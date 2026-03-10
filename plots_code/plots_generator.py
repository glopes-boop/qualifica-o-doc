import numpy as np
import pandas as pd
import seaborn as sns
from gdrive_download import GetData
import root_pandas as rp
import matplotlib.pyplot as plt
from utils import transform
from glob import glob


class PlotGenerator:

    def __init__(self, plot_args, image_size=(1024, 768), format='pdf') -> None:
        self.plot_args = plot_args
        self.image_size = image_size
        self.format = format

    def get_data(self) -> None:
        file_ids = self.plot_args['ids']
        destination = self.plot_args['destinations']
        mode = self.plot_args['mode']
        for f, dest, mode in zip(file_ids, destination, mode):
            if mode == 'drive':
                print(f'/n Downloadig file id {f}')
                file_getter = GetData(f, dest)
                file_getter.download_file_from_google_drive()
                print('\n Done !!! \n')

    def plot_filter_f1(self):
        pallete = ['#272627', '#535253', '#515051',
                   '#717071', '#939293', '#B6B5B6', '#DAD9DA']
        filter_results_df = pd.read_csv(
            'plots_code/filter_results.csv', index_col=False)

        filter_results_df['particle'] = filter_results_df['path'].str.split(
            '_').str[0]
        filter_results_df['energy'] = filter_results_df['path'].str.split(
            '_').str[1].astype(int)
        filter_results_df['event'] = filter_results_df['path'].str.split(
            '_').str[4]
        filter_results_df['filter_name'] = filter_results_df['filter'].str.split('_').str[
            0]
        filter_results_df['parameter'] = filter_results_df['filter'].str.split('_').str[
            1]
        filter_results_df.drop('path', axis=1, inplace=True)
        filter_results_df['filter'] = filter_results_df['filter'].apply(
            lambda x: f"{x.split('_')[0]},p={x.split('_')[1]}")

        f, axes = plt.subplots(1, 2, figsize=(20, 15))
        sns.stripplot(x="filter", y="best_f1",
                      data=filter_results_df, ax=axes[0], palette=[pallete[0]] + 3*pallete + [pallete[0]])
        sns.boxplot(x="filter", y="best_f1",
                    data=filter_results_df, ax=axes[1], palette=[pallete[0]] + 3*pallete + [pallete[0]])
        for ax in axes:
            ax.grid(True)
            ax.set_ylabel('Best F1-Score', fontsize=12)
            ax.set_xlabel('', fontsize=12)
        for ax in f.axes:
            plt.sca(ax)
            plt.xticks(rotation=90, fontsize=12)
            plt.yticks(fontsize=12)

        # plt.show()
        plt.savefig('image_files/f1_all.pdf', format='pdf')
        plt.close('all')

        f, axes = plt.subplots(1, 2, figsize=(20, 15))
        sns.stripplot(x="filter", y="best_f1",
                      data=filter_results_df, ax=axes[0], hue='particle', palette=['#515051', '#B6B5B6'], marker='o')
        sns.boxplot(x="filter", y="best_f1",
                    data=filter_results_df, ax=axes[1], hue='particle', palette=['#515051', '#B6B5B6'])
        for ax in axes:
            ax.grid(True)
            ax.set_ylabel('Best F1-Score', fontsize=12)
            ax.set_xlabel('', fontsize=12)
        for ax in f.axes:
            plt.sca(ax)
            plt.xticks(rotation=90, fontsize=12)
            plt.yticks(fontsize=12)

        plt.savefig('image_files/f1_particle.pdf', format='pdf')
        plt.close('all')

        f, axes = plt.subplots(1, 2, figsize=(20, 15))
        sns.stripplot(x="filter", y="best_f1",
                      data=filter_results_df, ax=axes[0], hue='energy', marker='o', palette=pallete[:-1])
        sns.boxplot(x="filter", y="best_f1",
                    data=filter_results_df, ax=axes[1], hue='energy', palette=pallete[:-1])
        for ax in axes:
            ax.grid(True)
            ax.set_ylabel('Best F1-Score', fontsize=12)
            ax.set_xlabel('', fontsize=12)
        for ax in f.axes:
            plt.sca(ax)
            plt.xticks(rotation=90, fontsize=12)
            plt.yticks(fontsize=12)

        plt.savefig('image_files/f1_energy.pdf', format='pdf')
        plt.close('all')

        print('ok')

    def simulation_clustering_data_gen(self):
        dir_prefix = self.plot_args['clustering_path']
        files = glob(dir_prefix + '*.root')
        merged_data = []
        for file_name in files:
            try:
                transformed_data = transform(
                    file_name, radius=400, it=2, cl_integral=1000000, n_hits=30)
                merged_data.append(transformed_data)
            except Exception as e:
                print('Error at file {}: {}'.format(file_name, e))
        all_data = pd.concat(merged_data, axis=0)
        run_map = pd.DataFrame({
            "run": [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010, 8011, 8012],
            "energy": [60, 30, 6, 30, 10, 1, 60, 6, 3, 3, 1, 100, 10],
            "particle": ['He', 'He', 'ER', 'ER', 'ER', 'He', 'ER', 'He', 'He', 'ER', 'ER', 'He', 'He']
        })

        all_data_include_info = pd.merge(all_data, run_map, on='run')
        all_data_include_info['n_pts'] = all_data_include_info['n_pts'].astype(
            int)

        en = 60
        truth_en = en*1000/2
        ax = sns.catplot(x="filter_name", y="cl_integral",
                         data=all_data_include_info[all_data_include_info['energy'] == en], kind="box", hue="n_pts",
                         height=8, aspect=1.5, cat='particle')
        ax = sns.lineplot(x="filter_name", y="cl_integral",
                          data=pd.DataFrame({
                              "cl_integral": [truth_en, truth_en, truth_en, truth_en, truth_en],
                              "filter_name": ["cygno", "mean", "median", "unet", "gaussian"],

                          }))

        plt.show()


if __name__ == '__main__':

    plot_generator = PlotGenerator(plot_args={'ids': ['filter_results.csv'],
                                              'destinations': ['./filter_results.csv'],
                                              'mode': ['local'],
                                              'clustering_path': 'clustering_results/'})
    plot_generator.get_data()
    plot_generator.plot_filter_f1()
    plot_generator.simulation_clustering_data_gen()
