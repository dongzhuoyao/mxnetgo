# Author: Tao Hu <taohu620@gmail.com>
import os
import logger
def JSONWriter(mod, prefix, period=1, save_optimizer_states=False):
    """
    Write all scalar data to a json file under ``logger.get_logger_dir()``, grouped by their global step.
    This monitor also attemps to recover the epoch number during setup,
    if an existing json file is found at the same place.
    """

    FILENAME = 'stat.json'
    """
    The name of the json file.
    """

    period = int(max(1, period))

    _dir = logger.get_logger_dir()
    _fname = os.path.join(_dir, FILENAME)


    # pylint: disable=unused-argument
    def _callback(iter_no, sym=None, arg=None, aux=None):
        """The checkpoint function."""
        if (iter_no + 1) % period == 0:
            mod.save_checkpoint(prefix, iter_no + 1, save_optimizer_states)

    return _callback


    def _before_train(self):
        self._dir = logger.get_logger_dir()
        self._fname = os.path.join(self._dir, self.FILENAME)

        if os.path.isfile(self._fname):
            logger.info("Found existing JSON at {}, will append to it.".format(self._fname))
            with open(self._fname) as f:
                self._stats = json.load(f)
                assert isinstance(self._stats, list), type(self._stats)

            try:
                epoch = self._stats[-1]['epoch_num'] + 1
            except Exception:
                pass
            else:
                # TODO is this a good idea?
                logger.info("Found training history from JSON, now starting from epoch number {}.".format(epoch))
                self.trainer.loop.starting_epoch = epoch
                self.trainer.loop._epoch_num = epoch - 1
        else:
            self._stats = []
        self._stat_now = {}

        self._last_gs = -1

    def _trigger_step(self):
        # will do this in trigger_epoch
        if self.local_step != self.trainer.steps_per_epoch - 1:
            self._push()

    def _trigger_epoch(self):
        self._push()

    def process_scalar(self, name, val):
        self._stat_now[name] = val

    def _push(self):
        """ Note that this method is idempotent"""
        if len(self._stat_now):
            self._stat_now['epoch_num'] = self.epoch_num
            self._stat_now['global_step'] = self.global_step

            self._stats.append(self._stat_now)
            self._stat_now = {}
            self._write_stat()

    def _write_stat(self):
        tmp_filename = self._fname + '.tmp'
        try:
            with open(tmp_filename, 'w') as f:
                json.dump(self._stats, f)
            shutil.move(tmp_filename, self._fname)
        except IOError:  # disk error sometimes..
            logger.exception("Exception in JSONWriter._write_stat()!")