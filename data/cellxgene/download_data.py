
import cellxgene_census
import os
import argparse
import traceback
import gc
import random
import re
import shutil
import warnings
import scanpy as sc
import numpy as np
from anndata import AnnData
from pathlib import Path
from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from typing_extensions import Self, Literal
from data_utils import GeneVocab, VALUE_FILTER, VERSION, DataBank
import pandas as pd
import os
import h5py
from pathlib import Path
# Predefined queries and settings
MAX_PARTITION_SIZE = 200000
queries = ["heart", "blood", "brain", "lung", "kidney",
    "intestine", "pancreas", "others", "pan-cancer"]
# queries = ["intestine"]
main_table_key = "counts"
token_col = "feature_name"
random.seed(42)
def retrieve_soma_idx(query_name: str) -> List[str]:
    """Retrieve cell soma ids based on the query name from the cellxgene census."""
    with cellxgene_census.open_soma(census_version=VERSION) as census:
        cell_metadata = census["census_data"]["homo_sapiens"].obs.read(
            value_filter=VALUE_FILTER[query_name],
            column_names=["soma_joinid"]
        )
        cell_metadata = cell_metadata.concat()
        cell_metadata = cell_metadata.to_pandas()
        return cell_metadata["soma_joinid"].tolist()

def convert2file(idx_list: List[str], query_name: str, output_dir: str) -> None:
    """Convert the retrieved index list to a file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, f"{query_name}.idx")
    with open(file_path, 'w') as file:
        for item in idx_list:
            file.write(f"{item}\n")

def build_soma_idx(query_name: str, output_dir: str) -> None:
    """Build the soma index for cells under the specified query name."""
    file_path = os.path.join(output_dir, f"{query_name}.idx")
    if not os.path.exists(file_path):  # Check if file exists before retrieving and converting
        idx_list = retrieve_soma_idx(query_name)
        convert2file(idx_list, query_name, output_dir)
    else:
        print(f"Skipping query as index file already exists: {file_path}")

def load_index_list(query_name: str, index_dir: str) -> List[int]:
    """Load the index list from a file."""
    file_path = os.path.join(index_dir, f"{query_name}.idx")
    with open(file_path, 'r') as fp:
        idx_list = [int(line.strip()) for line in fp]
    return idx_list

def partition_list(id_list: List[int], partition_size: int) -> List[List[int]]:
    """Partition the index list into chunks."""
    return [id_list[i:i + partition_size] for i in range(0, len(id_list), partition_size)]
def download_and_save_data(partition: List[int], query_name: str, output_dir: str, index_dir: str, partition_idx: int):
    """Download and save data for a given partition only if the file doesn't already exist."""
    query_dir = os.path.join(output_dir, query_name)
    os.makedirs(query_dir, exist_ok=True)
    output_file = os.path.join(query_dir, f"partition_{partition_idx}.h5ad")

    if not os.path.exists(output_file):  # Check if file already exists
        with cellxgene_census.open_soma(census_version=VERSION) as census:
            adata = cellxgene_census.get_anndata(census,
                                                 organism="Homo sapiens",
                                                 obs_coords=partition)
        adata.write_h5ad(output_file)
        print(f"Downloaded and saved: {output_file}")
    else:
        print(f"File already exists, skipping download: {output_file}")


def preprocess(adata, N=10000):
    """Preprocess the data for scBank."""
    sc.pp.filter_genes(adata, min_counts=(3 / 10000) * N)
    adata.layers['counts'] = adata.X.copy()
    return adata
def build_scbank(input_dir: str, output_dir: str, vocab_file: str, N: int):
    """Convert AnnData objects to scBank format. Skips processing if output exists."""
    input_path = Path(input_dir)
    scoutput_path = Path(output_dir)
    vocab = GeneVocab.from_file(Path(vocab_file))
    files = list(input_path.glob("*.h5ad"))

    for f in files:
        output_subdir = scoutput_path / f"{f.stem}.scb"
        if output_subdir.exists():
            print(f"Skipping {f.stem} as output already exists.")
            continue  # Skip processing this file if the output directory already exists

        try:
            adata = sc.read_h5ad(f)
            adata = preprocess(adata, N)
            db = DataBank.from_anndata(
                adata,
                vocab=vocab,
                to=scoutput_path / f"{f.stem}.scb",
                main_table_key=main_table_key,
                token_col=token_col,
                immediate_save=False,
            )
            db.meta_info.on_disk_format = "parquet"
            db.sync()
            del adata, db
            gc.collect()
        except Exception as e:
            traceback.print_exc()
            warnings.warn(f"Failed to process {f.name}: {e}")
            shutil.rmtree(output_subdir, ignore_errors=True)  # Ensure clean-up if processing fails


def create_symbolic_links(source_dir, link_dir):
    # Ensure the link directory exists
    link_dir_path = Path(source_dir) / link_dir
    os.makedirs(link_dir_path, exist_ok=True)

    pattern = re.compile(r'partition_(\d+)\.scb')  # Regex to extract digits from the partition name

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file == 'counts.datatable.parquet':
                full_path = Path(root) / file
                parts = Path(root).parts
                organ_name = parts[1]  # Now the organ name is at position -3 considering the structure
                partition_folder = parts[-1]
                match = pattern.match(partition_folder)
                if match:
                    partition_id = match.group(1)  # Extracted numeric part
                    link_name = f"{organ_name}_partition_{partition_id}_{file}"
                    full_link_path = link_dir_path / link_name
                    # relative_path = full_path.relative_to(link_dir_path.parent)
                    relative_path = Path('..') / full_path.relative_to(link_dir_path.parent)
                    if not full_link_path.exists():
                        full_link_path.symlink_to(relative_path)
                        print(f"Link created for {full_link_path}")
                    else:
                        print(f"Link already exists for {full_link_path}")
                else:
                    print(f"No valid partition number found in {partition_folder}")



def split_data(base_dir, link_dir):
    source_dir = os.path.join(base_dir, link_dir)
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    files = [f for f in os.listdir(source_dir) if f.endswith('.datatable.parquet')]
    random.shuffle(files)
    total_files = len(files)
    train_cutoff = int(total_files * 0.9)
    train_files = files[:train_cutoff]
    test_files = files[train_cutoff:]
    def link_files(file_list, target_dir):
        for file in file_list:
            target_path = os.path.join(target_dir, file)
            source_path = os.path.join("..", link_dir, file)
            if not os.path.islink(target_path):
                os.symlink(source_path, target_path)

    link_files(train_files, train_dir)
    link_files(test_files, test_dir)




def combine_and_sample(base_dir, dir_names):
    for dir_name in dir_names:
        dir_path = Path(base_dir) / dir_name
        if not dir_path.exists():
            print(f"Directory {dir_path} does not exist.")
            continue
        mini_file_path = dir_path / f"{dir_name}_mini_data.parquet"
        if mini_file_path.exists():
            print(f"File '{mini_file_path}' already exists, skipping.")
            continue

        parquet_files = list(dir_path.glob("*.datatable.parquet"))
        batch_size = 20
        batch_number = 1

        def convert_series_of_arrays(series, dtype):
            """Convert arrays within a pandas series to a specified dtype and check for overflow."""
            def safe_convert(array):
                temp_array = array.astype(dtype)
                if dtype == np.float16 and np.isinf(temp_array).any():
                    return array.astype(np.float32)
                return temp_array
            return series.apply(safe_convert)


        for i in range(0, len(parquet_files), batch_size):
            df_list = []
            combined_file_path = dir_path / f"{dir_name}_combined_data_{batch_number}.parquet"
            if combined_file_path.exists():
                print(f"File '{combined_file_path}' already exists, skipping.")
                batch_number += 1
                continue
            batch_files = parquet_files[i:i + batch_size]
            for file in batch_files:
                print(f"Processing file: {file}")
                df = pd.read_parquet(file)

                df['genes'] = df['genes'].apply(lambda array: array.astype(np.float32))
                df['expressions'] = df['expressions'].apply(lambda array: np.log1p(array.astype(np.float32)))
                all_expressions = np.concatenate(df['expressions'].values)
                max_value = all_expressions.max()
                min_value = all_expressions.min()
                print(f"Maximum value in 'expressions': {max_value}")
                print(f"Minimum value in 'expressions': {min_value}")
                df.drop('id', axis=1, inplace=True)
                df_list.append(df)

            combined_df = pd.concat(df_list, ignore_index=True)
            combined_df.to_parquet(str(combined_file_path))
            print(f"Combined file saved to: {combined_file_path}")

            mini_df = combined_df.sample(frac=0.1, random_state=42)
            mini_file_path = dir_path / f"{dir_name}_mini_data_{batch_number}.parquet"
            mini_df.to_parquet(str(mini_file_path))
            print(f"Mini dataset file saved to: {mini_file_path}")
            very_mini_df = mini_df.sample(frac=0.2, random_state=42)
            very_mini_file_path = dir_path / f"{dir_name}_very_mini_data_{batch_number}.parquet"
            very_mini_df.to_parquet(str(very_mini_file_path))
            print(f"Very mini dataset file saved to: {very_mini_file_path}")

            batch_number += 1
            del combined_df
            del mini_df
            del very_mini_df
            del df_list
            del df

        vy_mini_files = list(dir_path.glob(f"{dir_name}_very_mini_data_*.parquet"))
        mini_df_list = [pd.read_parquet(file) for file in vy_mini_files]
        final_mini_df = pd.concat(mini_df_list, ignore_index=True)
        final_mini_path = dir_path / f"{dir_name}_very_mini_data.parquet"
        final_mini_df.to_parquet(str(final_mini_path))
        print(f"Final mini dataset file saved to: {final_mini_path}")

        mini_files = list(dir_path.glob(f"{dir_name}_mini_data_*.parquet"))
        mini_df_list = [pd.read_parquet(file) for file in mini_files]
        final_mini_df = pd.concat(mini_df_list, ignore_index=True)
        final_mini_path = dir_path / f"{dir_name}_mini_data.parquet"
        final_mini_df.to_parquet(str(final_mini_path))
        print(f"Final mini dataset file saved to: {final_mini_path}")

        for file in mini_files:
            file.unlink()
            print(f"Deleted: {file}")
        for file in vy_mini_files:
            file.unlink()
            print(f"Deleted: {file}")


def main():
    parser = argparse.ArgumentParser(description='Build soma index list based on predefined query list')
    parser.add_argument("--index-dir", type=str, required=True, help="Directory to store the output idx files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store the downloaded data")
    parser.add_argument("--vocab-file", type=str, required=True, help="File containing the gene vocabulary")
    parser.add_argument("--scoutput-path", type=str, required=True, help="Directory to save scBank data")
    parser.add_argument("--link-dir", type=str, required=True, help="Directory to save scBank data")
    parser.add_argument("--N", type=int, default=200000, help="Num to save scBank data")
    args = parser.parse_args()
    for query_name in queries:
        print(f"Processing query: {query_name}")
        build_soma_idx(query_name, args.index_dir)  # corrected the attribute name here

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    for query_name in queries:
        print(f"Downloading query: {query_name}")
        idx_list = load_index_list(query_name, args.index_dir)
        partitions = partition_list(idx_list, MAX_PARTITION_SIZE)
        for i, partition in enumerate(partitions):
            download_and_save_data(partition, query_name, args.output_dir, args.index_dir, i)

    ##generate scdata
    if not os.path.exists(args.scoutput_path):
        os.makedirs(args.scoutput_path, exist_ok=True)
    for query_name in queries:
        print(f"Processing to scbankdata {query_name}")
        input_dir = os.path.join(args.output_dir, query_name)
        output_dir = os.path.join(args.scoutput_path, query_name)
        os.makedirs(output_dir, exist_ok=True)
        build_scbank(input_dir, output_dir, args.vocab_file, args.N)

    create_symbolic_links(args.scoutput_path, args.link_dir)
    if not os.path.exists(os.path.join(args.scoutput_path, 'train2')):
        split_data(args.scoutput_path, args.link_dir)


    directories = ["test2", "train2"]
    combine_and_sample(args.scoutput_path, directories)

if __name__ == "__main__":
    main()

# python download_data.py --index-dir ./idx_dir --output-dir ./data_raw --vocab-file vocab.json --scoutput-path ./data_scbank --link-dir ./all_counts



