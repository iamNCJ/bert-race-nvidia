import sys
import os
import torch.distributed as dist


def init_print(rank, size, debug_print=True):
    if not debug_print:
        """ In case run on hundreds of nodes, you may want to mute all the nodes except master """
        if rank > 0:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
    else:
        # labelled print with info of [rank/size]
        old_out = sys.stdout

        class LabeledStdout:
            def __init__(self, rank, size):
                self._r = rank
                self._s = size
                self.flush = sys.stdout.flush

            def write(self, x):
                if x == '\n':
                    old_out.write(x)
                else:
                    old_out.write('[%d/%d] %s' % (self._r, self._s, x))

        sys.stdout = LabeledStdout(rank, size)


if __name__ == "__main__":
    dist.init_process_group(backend='mpi')
    size = dist.get_world_size()
    rank = dist.get_rank()
    init_print(rank, size)
