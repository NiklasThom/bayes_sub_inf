import os
from tests.qwen_data.helper_scripts.save_datasets_and_params import save_dataset_as_numpy

script_dir = os.path.abspath("tests/qwen_data/helper_scripts")

save_dataset_as_numpy(
    dataset_name="ag_news",
    model_name="Qwen/Qwen2.5-0.5B",
    max_seq_len=128,
    batch_size=1,
    output_dir=os.path.join(script_dir, "data_store", "datasets"),
    save_wandb_artifact=False,
    wandb_project=None,
    script_dir=script_dir,
)
