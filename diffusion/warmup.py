from expertpara.diep import ep_cache_clear, ep_cached_tensors_size, ep_cache_init,ep_to_vc,ep_to_vu
from seqpara.df import sp_cache_init, sp_cache_clear, sp_cached_tensors_size,sp_to_vc,sp_to_vu

def ep_requireSync(current_step,para_mode,):
    '''
        current_step: the step count start from 0;
        para_mode: teh class models.ParaMode; store the prarameters of DiT
    '''
    if para_mode.ep_async_warm_up < 1 and para_mode.strided_sync < 1:
        return
    
    need_clear = False

    if para_mode is not None:
        start_step = current_step - para_mode.ep_async_warm_up
        if start_step < 0:
            need_clear = True
        elif para_mode.strided_sync > 0:
            if (start_step + 1) % para_mode.strided_sync == 0:
                need_clear = True
    if need_clear:
        ep_cache_clear()
    

def sp_requireSync(current_step,para_mode):
    if current_step < para_mode.sp_async_warm_up:
        sp_cache_clear()