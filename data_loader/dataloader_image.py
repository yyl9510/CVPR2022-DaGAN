import torch
import torch.utils.data as data

from torch.utils.data.distributed import DistributedSampler

from .vfhq_dataset import VFHQDataset


def get_dataloader(
    dataset_name,
    split,
    batch_size,
    is_train,
    is_distribute=True,
    drop_last=True,
    info_logger=None,
    size=256,
    num_workers=8,
    cube_config=None,
    **kwargs,
):
    if dataset_name == "vfhq":
        dataset = VFHQDataset(
            split=split,
            is_train=is_train,
            size=size,
            **kwargs,
        )

    else:
        raise NotImplementedError

    dataset_size = len(dataset)
    if is_distribute:
        if cube_config is not None:
            sampler = DistributedSampler(
                dataset=dataset,
                num_replicas=cube_config["runtime_ngpus"] // cube_config["plan_ngpus"],
                rank=torch.distributed.get_rank() // cube_config["plan_ngpus"],
                shuffle=True,
            )
            loader = data.DataLoader(
                dataset=dataset, batch_size=batch_size * cube_config["plan_ngpus"], sampler=sampler, num_workers=num_workers, drop_last=drop_last
            )
        else:
            sampler = DistributedSampler(dataset=dataset, shuffle=True)
            loader = data.DataLoader(
                dataset=dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=drop_last
            )
    else:
        loader = data.DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last
        )
    loader_length = len(loader)

    if info_logger:
        info_logger.info(f"Length of {split} dataset: {dataset_size} dataloader: {loader_length}")

    return loader, dataset_size, loader_length


if __name__ == "__main__":
    import os

    from torchvision.utils import save_image

    loader, dataset_size, loader_length = get_dataloader(
        root_dir="./dataset_root/VFHQ_datasets_extracted_example",
        dataset_name="vfhq",
        data_type="two",
        split="test",
        is_train=True,
        batch_size=1,
        debug=True,
        is_distribute=False,
        cube_config=None,
    )

    for i, data_batch in enumerate(loader):
        for k, v in data_batch.items():
            if torch.is_tensor(v):
                print(k, v.shape, v.min(), v.max())
            else:
                print(k, v)

        break
