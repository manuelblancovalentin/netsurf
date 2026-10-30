"""Bridge existing exact-engine counters to the existing single Rich Live queue."""
from .progress import ExperimentQueueItem, HierarchicalProgressReporter, ProgressEvent


class CampaignProgress:
    def __init__(self, names, *, enabled=True, planned=1):
        self.names = names
        self.seen = set()
        self.reporter = (HierarchicalProgressReporter([
            ExperimentQueueItem(name, "frozen campaign", {"setup": 1, "authoritative states": planned, "finalization": 1})
            for name in names]) if enabled else None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.reporter:
            self.reporter.finish()

    def start(self, index):
        if self.reporter:
            self.reporter.start_experiment(index)

    def callback(self, index):
        def emit(stage, completed, total):
            if self.reporter:
                first = index not in self.seen
                self.seen.add(index)
                if first:
                    # Attribute initial asset verification/model loading to one
                    # setup operation, not to every future inference state. The
                    # first callback can already report several completed states.
                    self.reporter.update(index, ProgressEvent(self.names[index], "Setup complete", 1, total + 2,
                        work_class="setup", remaining_work=(("authoritative states", total), ("finalization", 1))))
                self.reporter.update(index, ProgressEvent(self.names[index], stage, completed + 1, total + 2,
                    work_class="initial state batch" if first else "authoritative states",
                    remaining_work=(("authoritative states", total - completed), ("finalization", 1))))
        return emit

    def complete(self, index, *, resumed=False):
        if self.reporter:
            if resumed:
                self.reporter.start_experiment(index)
            self.reporter.complete_experiment(index, "Verified existing result" if resumed else "Checksums verified")

    def fail(self, index, error):
        if self.reporter:
            self.reporter.fail_experiment(index, error)
