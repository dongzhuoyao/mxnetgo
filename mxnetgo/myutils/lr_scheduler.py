__all__ = ['WarmupMultiFactorScheduler', 'StepScheduler']
from mxnet.lr_scheduler import LRScheduler
from . import logger
class WarmupMultiFactorScheduler(LRScheduler):
    """Reduce learning rate in factor at steps specified in a list

    Assume the weight has been updated by n times, then the learning rate will
    be

    base_lr * factor^(sum((step/n)<=1)) # step is an array

    Parameters
    ----------
    step: list of int
        schedule learning rate after n updates
    factor: float
        the factor for reducing the learning rate
    """
    def __init__(self, step, factor=1, warmup=False, warmup_lr=0, warmup_step=0):
        super(WarmupMultiFactorScheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0
        self.warmup = warmup
        self.warmup_lr = warmup_lr
        self.warmup_step = warmup_step

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """

        # NOTE: use while rather than if  (for continuing training via load_epoch)
        if self.warmup and num_update < self.warmup_step:
            return self.warmup_lr
        while self.cur_step_ind <= len(self.step)-1:
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= self.factor
                #logging.info("Update[%d]: Change learning rate to %0.5e",
                #             num_update, self.base_lr)
            else:
                return self.base_lr
        return self.base_lr


class StepScheduler(LRScheduler):
    """Reduce the learning rate by given a list of steps.

        Assume there exists *k* such that::

           step[k] <= num_update and num_update < step[k+1]

        Then calculate the new learning rate by::

           base_lr * pow(factor, k+1)

        Parameters
        ----------
        step: list of int
            The list of steps to schedule a change
        factor: float
            The factor to change the learning rate.
        """

    def __init__(self, epoch_update_nums,steplist, factor=1):
        super(StepScheduler, self).__init__()
        assert isinstance(steplist, list) and len(steplist) >= 1
        for i, _step in enumerate(steplist):
            if i != 0 and steplist[i] <= steplist[i - 1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        if factor > 1.0:
            raise ValueError("Factor must be no more than 1 to make lr reduce")
        self.steplist = steplist
        self.epoch_update_nums = epoch_update_nums
        self.cur_lr = steplist[0][1]

    def __call__(self, num_update):
        #if num_update%self.epoch_update_nums == 0:
        #    logger.info("cur lr: {}".format(self.cur_lr))
        for (e,v) in self.steplist:
            if num_update < self.epoch_update_nums*e:
                self.cur_lr = v
                return self.cur_lr

        return self.cur_lr
