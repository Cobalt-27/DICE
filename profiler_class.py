import os
import torch


class ProfileExp:
    def __init__(self, save_dir, prefix='step', init_step=0,
                 wait=0, warmup=2, active=1, repeat=1, profile_at=-10000):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.prefix = prefix
        self.profile_at = profile_at

        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.num_steps = (self.wait + self.warmup + self.active) * self.repeat
        self.remain = self.num_steps

        self.disable = profile_at is None or profile_at < 0
        # assert self.disable or (self.profile_at > self.remain), f"Expect {self.profile_at} > {self.remain} for proper warmup."
        if not self.disable:
            print("Valid Profiler")
        self.logging_step = init_step
        self.prof = None
        if not self.disable and (self.logging_step + self.num_steps) % self.profile_at == 0:
            self._start()

    def step(self):
        if self.disable:
            return
        self.logging_step += 1

        if self.prof is not None:
            if self.remain == 0:
                self._finalize()
            else:
                self.prof.step()
                self.remain -= 1

        if (self.logging_step + self.num_steps) % self.profile_at == 0:
            self._start()

    def _start(self):
        self.remain = self.num_steps
        self.prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=self.wait, warmup=self.warmup, active=self.active, repeat=self.repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(self.save_dir, f'{self.prefix}_{self.logging_step + self.num_steps}')),
            record_shapes=True,
            with_stack=True,
            profile_memory=True
        )
        self.prof.start()

    def _finalize(self):
        self.prof.__exit__(None, None, None)
        self.prof = None
        self.disable = True

