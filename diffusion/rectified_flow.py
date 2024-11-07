import torch 
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist 
from expertpara.diep import ep_set_step, ep_async_pipeline_enabled, ep_cache_put_on_miss
from .warmup import ep_forced_sync,sp_require_sync

class RectifiedFlow(torch.nn.Module):
    def __init__(self, model, ln=True):
        super().__init__()
        self.model = model
        self.ln = ln
        self.stratified = False 
        if isinstance(model, DDP):
            self.learn_sigma = model.module.learn_sigma 
        else:
            self.learn_sigma = model.learn_sigma 

        
    def forward(self, x, cond):

        b = x.size(0)
        if self.ln:
            if self.stratified:
                # stratified sampling of normals
                # first stratified sample from uniform
                quantiles = torch.linspace(0, 1, b + 1).to(x.device)
                z = quantiles[:-1] + torch.rand((b,)).to(x.device) / b
                # now transform to normal
                z = torch.erfinv(2 * z - 1) * math.sqrt(2)
                t = torch.sigmoid(z)
            else:
                nt = torch.randn((b,)).to(x.device)
                t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1

        # make t, zt into same dtype as x
        zt, t = zt.to(x.dtype), t.to(x.dtype)
        vtheta = self.model(zt, t, cond) 
        if self.learn_sigma == True: 
            vtheta, _ = vtheta.chunk(2, dim=1) 
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), {"batchwise_loss": ttloss}

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        raise RuntimeError("shouldn't reach here")
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond) 
            if self.learn_sigma == True: 
                vc, _ = vc.chunk(2, dim=1) 
            if null_cond is not None:
                vu = self.model(z, t, null_cond) 
                if self.learn_sigma == True: 
                    vu, _ = vu.chunk(2, dim=1) 
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images

    @torch.no_grad()
    def sample_with_xps(self, z, cond, null_cond=None, sample_steps=50, 
                        cfg=2.0,para_mode = None):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        # XXX: we only need to store the last image
        # images = [z]
        final_image = z
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            
            diep_forced_sync = False
            if para_mode is not None:
                if para_mode.ep_async:
                    ep_set_step(i)
                    diep_forced_sync = ep_forced_sync(i, sample_steps, para_mode)
                if para_mode.sp_async:
                    sp_require_sync(i, sample_steps, para_mode)
            

            # ep_cache_clear()
            assert self.learn_sigma
            if null_cond is not None:        
                # ep_cache_clear()
                from expertpara.diep import diep_force_sync, diep_cancel_sync, ep_cache_clear
                if diep_forced_sync:
                    assert para_mode is not None and para_mode.ep_async
                    diep_force_sync()
                    ep_cache_clear()
                if para_mode is not None and para_mode.ep_async and ep_async_pipeline_enabled():
                    # perform 2 fwd (cond/uncond)
                    
                    """
                    NOTE:
                    two micro batches are pipelined
                    have to carefully manage how caches are updated on synced warmup steps
                    otherwise two microbatches may "pollute" each other's cache
                    """
                    ep_cache_put_on_miss(False, True)
                    vc = self.model(z, t, cond)
                    ep_cache_put_on_miss(True, False)
                    vu = self.model(z, t, null_cond)
                else:
                    merged_z = torch.cat((z, z), dim=0)
                    merged_t = torch.cat((t, t), dim=0)
                    merged_cond = torch.cat((cond, null_cond), dim=0)
                    v_merged = self.model(merged_z, merged_t, merged_cond)
                    vc, vu = torch.chunk(v_merged, 2, dim=0)
                
                if diep_forced_sync:
                    diep_cancel_sync()
                
                if self.learn_sigma == True: 
                    vc, _ = vc.chunk(2, dim=1)
                    vu, _ = vu.chunk(2, dim=1) 
                
                vc = vu + cfg * (vc - vu)
            else:
                vc = self.model(z, t, cond)
                if self.learn_sigma == True: 
                    vc, _ = vc.chunk(2, dim=1)  
            x = z - i * dt * vc
            z = z - dt * vc
            # images.append(x)
            final_image = x
        # return images
        return final_image