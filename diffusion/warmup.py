from expertpara.diep import ep_cache_clear
from seqpara.df import sp_cache_clear
from expertpara.diep import ep_cache_clear
from seqpara.df import sp_cache_clear

def ep_requireSync(i, sample_steps,para_mode,):
    '''
        i: sample_steps to 1
        para_mode: teh class models.ParaMode; store the prarameters of DiT
    '''
    if para_mode.ep_async_warm_up < 1 and para_mode.strided_sync < 1:
        return
    
    need_clear = False

    current_step_rev = sample_steps - i
    if current_step_rev <= para_mode.ep_async_warm_up and current_step_rev >0:
        need_clear=True
    elif para_mode.strided_sync > 0 and i % para_mode.strided_sync == 0:
        need_clear=True
    elif i <= para_mode.ep_async_cool_down:
        need_clear=True
        
    if need_clear:
        ep_cache_clear()
    

def sp_requireSync(i, sample_steps,para_mode):
    current_step_rev = sample_steps - i
    if current_step_rev <= para_mode.sp_async_warm_up and current_step_rev >0:
        sp_cache_clear()