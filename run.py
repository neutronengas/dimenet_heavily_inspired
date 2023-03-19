from notebooks.data_reader.data_container import DataContainer
from notebooks.data_reader.data_provider import DataProvider
from notebooks.data_reader.trainer import Trainer
#from .schedules import LinearWarmupExponentialDecay
from notebooks.data_reader.metrics import Metrics
from notebooks.dimenet_total.dimenet import DimeNet

ntrain = 1000
nvalid = 200

def __main__():
    model = DimeNet()
    data_container = DataContainer("./data/md17_aspirin.npz")
    data_provider = DataProvider(data_container=data_container, ntrain=ntrain, nvalid=nvalid)
    trainer = Trainer(model=model)
    metrics = Metrics("aspirin", ["E", "F"])
    train_set = data_provider.get_dataset("train")
    val_set = data_provider.get_dataset("val")
    test_set = data_provider.get_dataset("test")

    # pray
    trainer.train_on_batch(iter(train_set), metrics)
    trainer.test_on_batch(iter(val_set), metrics)
    return trainer.test_on_batch(iter(test_set), metrics)

__main__()