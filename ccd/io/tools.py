from .frame import Frame
from .frameset import FrameSet
from .info import FrameSetInfo

__all__ = ["frame_iter", "get_frame"]


def frame_iter(paths):
    """Loop over all frames in the files in `paths` regardless of single-/multiple-frame files."""
    for p in paths:
        try:
            fr = Frame.load_file(p)
            yield fr
        except ValueError:
            fs = FrameSet.load_file(p)
            for i in xrange(fs.num_frames):
                yield fs.get_frame(i, retplain=False)


def get_frame(framepath):
    """Get a frame from a single or multi-frame file."""
    tok = framepath.split(FrameSetInfo.framenumsep)
    if len(tok) == 1:
        return Frame.load_file(framepath)
    elif len(tok) == 2:
        path = tok[0]
        if tok[1].startswith("frame "):
            fnum = int(tok[1][6:])
        else:
            raise ValueError("Could not parse frame number from '%s': '%s'" % (tok[1], framepath))
        return FrameSet.load_file(path).get_frame(fnum, retplain=False)
    else:
        raise ValueError("Could not parse framepath: '%s'" % (framepath,))
