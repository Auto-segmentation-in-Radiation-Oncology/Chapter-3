import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


def draw_brace(ax, xspan, text, ypos=None):
    """Draws an annotated brace on the axes."""

    # Thanks to: https://stackoverflow.com/questions/18386210/annotating-ranges-of-data-in-matplotlib
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    if ypos is not None:
        ymin = ypos
    resolution = int(xspan/xax_span*100)*2+1  # guaranteed uneven
    beta = 300./xax_span  # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2+1)]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = ymin + (.05*y - .01)*yspan  # adjust vertical position

    ax.autoscale(False)
    ax.plot(x, y, color='black', lw=1)

    ax.text((xmax+xmin)/2., ymin+.07*yspan, text, ha='center', va='bottom')


def main():
    atlases = ['LCTSC-Train-S1-001',
               'LCTSC-Train-S1-002',
               'LCTSC-Train-S1-003',
               'LCTSC-Train-S1-004',
               'LCTSC-Train-S1-005',
               'LCTSC-Train-S1-006',
               'LCTSC-Train-S1-007',
               'LCTSC-Train-S1-008',
               'LCTSC-Train-S1-009',
               'LCTSC-Train-S1-010',
               'LCTSC-Train-S1-011',
               'LCTSC-Train-S1-012',
               'LCTSC-Train-S2-001',
               'LCTSC-Train-S2-002',
               'LCTSC-Train-S2-003',
               'LCTSC-Train-S2-004',
               'LCTSC-Train-S2-005',
               'LCTSC-Train-S2-006',
               'LCTSC-Train-S2-007',
               'LCTSC-Train-S2-008',
               'LCTSC-Train-S2-009',
               'LCTSC-Train-S2-010',
               'LCTSC-Train-S2-011',
               'LCTSC-Train-S2-012',
               'LCTSC-Train-S3-001',
               'LCTSC-Train-S3-002',
               'LCTSC-Train-S3-003',
               'LCTSC-Train-S3-004',
               'LCTSC-Train-S3-005',
               'LCTSC-Train-S3-006',
               'LCTSC-Train-S3-007',
               'LCTSC-Train-S3-008',
               'LCTSC-Train-S3-009',
               'LCTSC-Train-S3-010',
               'LCTSC-Train-S3-011',
               'LCTSC-Train-S3-012',
               'LCTSC-Test-S1-101',
               'LCTSC-Test-S1-102',
               'LCTSC-Test-S1-103',
               'LCTSC-Test-S1-104',
               'LCTSC-Test-S2-101',
               'LCTSC-Test-S2-102',
               'LCTSC-Test-S2-103',
               'LCTSC-Test-S2-104',
               'LCTSC-Test-S3-101',
               'LCTSC-Test-S3-102',
               'LCTSC-Test-S3-103',
               'LCTSC-Test-S3-104']

    patients = ['LCTSC-Test-S1-201',
                'LCTSC-Test-S1-202',
                'LCTSC-Test-S1-203',
                'LCTSC-Test-S1-204',
                'LCTSC-Test-S2-201',
                'LCTSC-Test-S2-202',
                'LCTSC-Test-S2-203',
                'LCTSC-Test-S2-204',
                'LCTSC-Test-S3-201',
                'LCTSC-Test-S3-202',
                'LCTSC-Test-S3-203',
                'LCTSC-Test-S3-204']

    organ_list = ['Heart', 'Lung_L', 'Lung_R', 'Esophagus', 'SpinalCord']

    output_root_dir = 'C:\\Mark\\Book\\Atlas selection code\\ResultsDataDemons\\'

    results_tables = []

    # load all the data into a big structure
    for patient in patients:
        print('Processing case: ' + patient)
        print('')
        results_for_patient = {'Heart': np.empty((0, 7), float),
                               'Lung_L': np.empty((0, 7), float),
                               'Lung_R': np.empty((0, 7), float),
                               'Esophagus': np.empty((0, 7), float),
                               'SpinalCord': np.empty((0, 7), float)}

        for atlas in atlases:

            result_csv = output_root_dir + patient + '\\' + atlas + '\\results.csv'
            if os.path.exists(result_csv):
                result_data = pd.read_csv(result_csv, header=0)

                for row_index in range(0, result_data.shape[0]):
                    row_data = result_data.iloc[row_index].values
                    results_for_patient[row_data[0]] = np.append(results_for_patient[row_data[0]],
                                                                 np.array([row_data[1:]]), axis=0)

        results_tables.append(results_for_patient)

    # plot performance for first patient for all organs
    results_for_patient = results_tables[0]
    for organ in organ_list:
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        data_to_plot = results_for_patient[organ]
        # sort by dice and plot
        data_to_plot = data_to_plot[data_to_plot[:, 0].argsort()[::-1]]
        oracleplt, = plt.plot(data_to_plot[:, 0], label='Oracle')
        plt.ylabel('DSC')
        plt.xlabel('Ranked atlas number')
        plt.title('Results for ' + organ)
        # sort by NMI rigid and plot
        data_to_plot = data_to_plot[data_to_plot[:, 1].argsort()[::-1]]
        nmirigidplt, = plt.plot(data_to_plot[:, 0], label='NMI Affine')
        # sort by NMI deformable and plot
        data_to_plot = data_to_plot[data_to_plot[:, 3].argsort()[::-1]]
        nmidefplt, = plt.plot(data_to_plot[:, 0], label='NMI Def')
        # sort by NMI local deformable and plot
        data_to_plot = data_to_plot[data_to_plot[:, 5].argsort()[::-1]]
        nmilocalplt, = plt.plot(data_to_plot[:, 0], label='NMI Local Def')
        # sort by RMSE rigid and plot
        data_to_plot = data_to_plot[data_to_plot[:, 2].argsort()]
        rmserigidplt, = plt.plot(data_to_plot[:, 0], label='RSME Affine')
        # sort by RMSE deformable and plot
        data_to_plot = data_to_plot[data_to_plot[:, 4].argsort()]
        rmsedefplt, = plt.plot(data_to_plot[:, 0], label='RMSE Def')
        # sort by RMSE local deformable and plot
        data_to_plot = data_to_plot[data_to_plot[:, 6].argsort()]
        rmselocalplt, = plt.plot(data_to_plot[:, 0], label='RMSE Local Def')
        plt.legend(handles=[oracleplt, nmirigidplt, nmidefplt, nmilocalplt, rmserigidplt, rmsedefplt, rmselocalplt])
        plt.savefig('C:\\Mark\\Book\\Atlas selection code\\Figures\\' + organ + '.png')

    # plot average performance for all patients for all organs
    sorted_by_dsc = np.empty((48, 12))
    sorted_by_nmirigid = np.empty((48, 12))
    sorted_by_nmidef = np.empty((48, 12))
    sorted_by_nmilocal = np.empty((48, 12))
    sorted_by_rmserigid = np.empty((48, 12))
    sorted_by_rmsedef = np.empty((48, 12))
    sorted_by_rmselocal = np.empty((48, 12))
    rank_by_dsc = np.empty((48, 12))
    rank_by_nmirigid = np.empty((48, 12))
    rank_by_nmidef = np.empty((48, 12))
    rank_by_nmilocal = np.empty((48, 12))
    rank_by_rmserigid = np.empty((48, 12))
    rank_by_rmsedef = np.empty((48, 12))
    rank_by_rmselocal = np.empty((48, 12))
    performance_for_organ = {'Heart': np.empty(48), 'Lung_L': np.empty(48), 'Lung_R': np.empty(48),
                             'Esophagus': np.empty(48), 'SpinalCord': np.empty(48)}
    data_for_rank_bar_plot = {'Heart': np.empty((8, 12)), 'Lung_L': np.empty((8, 12)), 'Lung_R': np.empty((8, 12)),
                              'Esophagus': np.empty((8, 12)), 'SpinalCord': np.empty((8, 12))}
    data_for_performance_bar_plot = {'Heart': np.empty((8, 12)), 'Lung_L': np.empty((8, 12)),
                                     'Lung_R': np.empty((8, 12)),
                                     'Esophagus': np.empty((8, 12)), 'SpinalCord': np.empty((8, 12))}
    for organ in organ_list:
        patient_number = 0
        for results_for_patient in results_tables:
            data_to_plot = results_for_patient[organ]
            # sort by dice
            data_to_plot = data_to_plot[data_to_plot[:, 0].argsort()[::-1]]
            # add rank column so we can get average rank
            data_to_plot = np.concatenate((data_to_plot, np.arange(1, 49)[:, None]), axis=1)
            sorted_by_dsc[:, patient_number] = data_to_plot[:, 0]
            rank_by_dsc[:, patient_number] = data_to_plot[:, 7]
            # sort by NMI rigid
            data_to_plot = data_to_plot[data_to_plot[:, 1].argsort()[::-1]]
            sorted_by_nmirigid[:, patient_number] = data_to_plot[:, 0]
            rank_by_nmirigid[:, patient_number] = data_to_plot[:, 7]
            # sort by NMI deformable
            data_to_plot = data_to_plot[data_to_plot[:, 3].argsort()[::-1]]
            sorted_by_nmidef[:, patient_number] = data_to_plot[:, 0]
            rank_by_nmidef[:, patient_number] = data_to_plot[:, 7]
            # sort by NMI local deformable
            data_to_plot = data_to_plot[data_to_plot[:, 5].argsort()[::-1]]
            sorted_by_nmilocal[:, patient_number] = data_to_plot[:, 0]
            rank_by_nmilocal[:, patient_number] = data_to_plot[:, 7]
            # sort by RMSE rigid
            data_to_plot = data_to_plot[data_to_plot[:, 2].argsort()]
            sorted_by_rmserigid[:, patient_number] = data_to_plot[:, 0]
            rank_by_rmserigid[:, patient_number] = data_to_plot[:, 7]
            # sort by RMSE deformable
            data_to_plot = data_to_plot[data_to_plot[:, 4].argsort()]
            sorted_by_rmsedef[:, patient_number] = data_to_plot[:, 0]
            rank_by_rmsedef[:, patient_number] = data_to_plot[:, 7]
            # sort by RMSE local deformable
            data_to_plot = data_to_plot[data_to_plot[:, 6].argsort()]
            sorted_by_rmselocal[:, patient_number] = data_to_plot[:, 0]
            rank_by_rmselocal[:, patient_number] = data_to_plot[:, 7]
            patient_number = patient_number + 1
        # plot for organ
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        oracleplt, = plt.plot(np.mean(sorted_by_dsc, axis=1), label='Oracle')
        performance_for_organ[organ] = np.mean(sorted_by_dsc, axis=1)  # used later for combined organ plot
        plt.ylabel('DSC')
        plt.xlabel('Ranked atlas number')
        plt.title('Average results for ' + organ)
        # sort by NMI rigid and plot
        nmirigidplt, = plt.plot(np.mean(sorted_by_nmirigid, axis=1), label='NMI Affine')
        # sort by NMI deformable and plot
        nmidefplt, = plt.plot(np.mean(sorted_by_nmidef, axis=1), label='NMI Def')
        # sort by NMI local deformable and plot
        nmilocalplt, = plt.plot(np.mean(sorted_by_nmilocal, axis=1), label='NMI Local Def')
        # sort by RMSE rigid and plot
        rmserigidplt, = plt.plot(np.mean(sorted_by_rmserigid, axis=1), label='RSME Affine')
        # sort by RMSE deformable and plot
        rmsedefplt, = plt.plot(np.mean(sorted_by_rmsedef, axis=1), label='RMSE Def')
        # sort by RMSE local deformable and plot
        rmselocalplt, = plt.plot(np.mean(sorted_by_rmselocal, axis=1), label='RMSE Local Def')
        plt.legend(handles=[oracleplt, nmirigidplt, nmidefplt, nmilocalplt, rmserigidplt, rmsedefplt, rmselocalplt])
        plt.savefig('C:\\Mark\\Book\\Atlas selection code\\Figures\\Average' + organ + '.png')

        # Aljabar style average plot for NMI
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        plt.scatter(np.arange(1, 49), np.mean(sorted_by_nmidef, axis=1))
        plt.ylabel('Average DSC over all test cases')
        plt.xlabel('Atlas rank by Global NMI after deformable registration')
        plt.title(organ)
        plt.savefig('C:\\Mark\\Book\\Atlas selection code\\Figures\\Aljabar average plot' + organ + '.png')

        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(8, 6)
        plt.scatter(np.arange(1, 49), sorted_by_nmidef[:, 0])
        plt.scatter(np.arange(1, 49), sorted_by_nmidef[:, 1])
        plt.scatter(np.arange(1, 49), sorted_by_nmidef[:, 2])
        plt.scatter(np.arange(1, 49), sorted_by_nmidef[:, 3])
        plt.scatter(np.arange(1, 49), sorted_by_nmidef[:, 4])
        plt.scatter(np.arange(1, 49), sorted_by_nmidef[:, 5])
        plt.scatter(np.arange(1, 49), sorted_by_nmidef[:, 6])
        plt.scatter(np.arange(1, 49), sorted_by_nmidef[:, 7])
        plt.scatter(np.arange(1, 49), sorted_by_nmidef[:, 8])
        plt.scatter(np.arange(1, 49), sorted_by_nmidef[:, 9])
        plt.scatter(np.arange(1, 49), sorted_by_nmidef[:, 10])
        plt.scatter(np.arange(1, 49), sorted_by_nmidef[:, 11])
        plt.ylabel('DSC')
        plt.xlabel('Atlas rank by Global NMI after deformable registration')
        plt.title(organ)
        plt.savefig('C:\\Mark\\Book\\Atlas selection code\\Figures\\Aljabar individual plot' + organ + '.png')

        # build random set
        # Get a random rank set for each patient
        # The use the same rank to look up performance
        # This will result is different plots each time it is run as it is random!
        rank_by_random = np.empty((10, 12))
        sorted_by_random = np.empty((10, 12))
        for i in range(0, 12):
            patient_random_rank = np.random.choice(np.arange(1, 49), size=10, replace=False)
            rank_by_random[:, i] = patient_random_rank
            sorted_by_random[:, i] = sorted_by_dsc[patient_random_rank - 1, i]

        data_for_rank_bar_plot[organ] = [np.mean(rank_by_dsc[0:10, :], axis=0),
                                         np.mean(rank_by_nmirigid[0:10, :], axis=0),
                                         np.mean(rank_by_nmidef[0:10, :], axis=0),
                                         np.mean(rank_by_nmilocal[0:10, :], axis=0),
                                         np.mean(rank_by_rmserigid[0:10, :], axis=0),
                                         np.mean(rank_by_rmsedef[0:10, :], axis=0),
                                         np.mean(rank_by_rmselocal[0:10, :], axis=0),
                                         np.mean(rank_by_random[0:10, :], axis=0)]
        data_for_performance_bar_plot[organ] = [np.mean(sorted_by_dsc[0:10, :], axis=0),
                                                np.mean(sorted_by_nmirigid[0:10, :], axis=0),
                                                np.mean(sorted_by_nmidef[0:10, :], axis=0),
                                                np.mean(sorted_by_nmilocal[0:10, :], axis=0),
                                                np.mean(sorted_by_rmserigid[0:10, :], axis=0),
                                                np.mean(sorted_by_rmsedef[0:10, :], axis=0),
                                                np.mean(sorted_by_rmselocal[0:10, :], axis=0),
                                                np.mean(sorted_by_random[0:10, :], axis=0)]

    # Plot rank plot
    oracle_mean = np.empty(5)
    nmirigid_mean = np.empty(5)
    nmidef_mean = np.empty(5)
    nmilocal_mean = np.empty(5)
    rmserigid_mean = np.empty(5)
    rmsedef_mean = np.empty(5)
    rmselocal_mean = np.empty(5)
    random_mean = np.empty(5)
    oracle_err = np.empty((2, 5))
    nmirigid_err = np.empty((2, 5))
    nmidef_err = np.empty((2, 5))
    nmilocal_err = np.empty((2, 5))
    rmserigid_err = np.empty((2, 5))
    rmsedef_err = np.empty((2, 5))
    rmselocal_err = np.empty((2, 5))
    random_err = np.empty((2, 5))
    organ_count = 0
    for organ in organ_list:
        oracle_mean[organ_count] = np.mean(data_for_rank_bar_plot[organ][0])
        oracle_err[:, organ_count] = [oracle_mean[organ_count] - np.min(data_for_rank_bar_plot[organ][0]),
                                      np.max(data_for_rank_bar_plot[organ][0]) - oracle_mean[organ_count]]
        nmirigid_mean[organ_count] = np.mean(data_for_rank_bar_plot[organ][1])
        nmirigid_err[:, organ_count] = [nmirigid_mean[organ_count] - np.min(data_for_rank_bar_plot[organ][1]),
                                        np.max(data_for_rank_bar_plot[organ][1]) - nmirigid_mean[organ_count]]
        nmidef_mean[organ_count] = np.mean(data_for_rank_bar_plot[organ][2])
        nmidef_err[:, organ_count] = [nmidef_mean[organ_count] - np.min(data_for_rank_bar_plot[organ][2]),
                                      np.max(data_for_rank_bar_plot[organ][2]) - nmidef_mean[organ_count]]
        nmilocal_mean[organ_count] = np.mean(data_for_rank_bar_plot[organ][3])
        nmilocal_err[:, organ_count] = [nmilocal_mean[organ_count] - np.min(data_for_rank_bar_plot[organ][3]),
                                        np.max(data_for_rank_bar_plot[organ][3]) - nmilocal_mean[organ_count]]
        rmserigid_mean[organ_count] = np.mean(data_for_rank_bar_plot[organ][4])
        rmserigid_err[:, organ_count] = [rmserigid_mean[organ_count] - np.min(data_for_rank_bar_plot[organ][4]),
                                         np.max(data_for_rank_bar_plot[organ][4]) - rmserigid_mean[organ_count]]
        rmsedef_mean[organ_count] = np.mean(data_for_rank_bar_plot[organ][5])
        rmsedef_err[:, organ_count] = [rmsedef_mean[organ_count] - np.min(data_for_rank_bar_plot[organ][5]),
                                       np.max(data_for_rank_bar_plot[organ][5]) - rmsedef_mean[organ_count]]
        rmselocal_mean[organ_count] = np.mean(data_for_rank_bar_plot[organ][6])
        rmselocal_err[:, organ_count] = [rmselocal_mean[organ_count] - np.min(data_for_rank_bar_plot[organ][6]),
                                         np.max(data_for_rank_bar_plot[organ][6]) - rmselocal_mean[organ_count]]
        random_mean[organ_count] = np.mean(data_for_rank_bar_plot[organ][7])
        random_err[:, organ_count] = [random_mean[organ_count] - np.min(data_for_rank_bar_plot[organ][7]),
                                      np.max(data_for_rank_bar_plot[organ][7]) - random_mean[organ_count]]
        organ_count = organ_count + 1

    x = np.arange(len(organ_list))  # the label locations
    width = 0.1  # the width of the bars
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    fig, ax = plt.subplots()
    ax.bar(x - 3.5 * width, oracle_mean, width, yerr=oracle_err, label='Oracle')
    ax.bar(x - 2.5 * width, nmirigid_mean, width, yerr=nmirigid_err, label='NMI Affine')
    ax.bar(x - 1.5 * width, nmidef_mean, width, yerr=nmidef_err, label='NMI Def')
    ax.bar(x - 0.5 * width, nmilocal_mean, width, yerr=nmilocal_err, label='NMI Local Def')
    ax.bar(x + 0.5 * width, rmserigid_mean, width, yerr=rmserigid_err, label='RMSE Affine')
    ax.bar(x + 1.5 * width, rmsedef_mean, width, yerr=rmsedef_err, label='RMSE Def')
    ax.bar(x + 2.5 * width, rmselocal_mean, width, yerr=rmselocal_err, label='RMSE Local Def')
    ax.bar(x + 3.5 * width, random_mean, width, yerr=random_err, label='Random')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average Rank over 12 test cases')
    ax.set_title('Average Rank of 10 selected atlases')
    ax.set_xticks(x)
    ax.set_xticklabels(organ_list)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
              borderaxespad=0, frameon=True, prop={'size': 8})

    fig.tight_layout()

    plt.savefig('C:\\Mark\\Book\\Atlas selection code\\Figures\\Rank bar plot.png')

    # Plot performance plot
    oracle_mean = np.empty(5)
    nmirigid_mean = np.empty(5)
    nmidef_mean = np.empty(5)
    nmilocal_mean = np.empty(5)
    rmserigid_mean = np.empty(5)
    rmsedef_mean = np.empty(5)
    rmselocal_mean = np.empty(5)
    random_mean = np.empty(5)
    oracle_err = np.empty((2, 5))
    nmirigid_err = np.empty((2, 5))
    nmidef_err = np.empty((2, 5))
    nmilocal_err = np.empty((2, 5))
    rmserigid_err = np.empty((2, 5))
    rmsedef_err = np.empty((2, 5))
    rmselocal_err = np.empty((2, 5))
    random_err = np.empty((2, 5))
    organ_count = 0
    for organ in organ_list:
        oracle_mean[organ_count] = np.mean(data_for_performance_bar_plot[organ][0])
        oracle_err[:, organ_count] = [oracle_mean[organ_count] - np.min(data_for_performance_bar_plot[organ][0]),
                                      np.max(data_for_performance_bar_plot[organ][0]) - oracle_mean[organ_count]]
        nmirigid_mean[organ_count] = np.mean(data_for_performance_bar_plot[organ][1])
        nmirigid_err[:, organ_count] = [nmirigid_mean[organ_count] - np.min(data_for_performance_bar_plot[organ][1]),
                                        np.max(data_for_performance_bar_plot[organ][1]) - nmirigid_mean[organ_count]]
        nmidef_mean[organ_count] = np.mean(data_for_performance_bar_plot[organ][2])
        nmidef_err[:, organ_count] = [nmidef_mean[organ_count] - np.min(data_for_performance_bar_plot[organ][2]),
                                      np.max(data_for_performance_bar_plot[organ][2]) - nmidef_mean[organ_count]]
        nmilocal_mean[organ_count] = np.mean(data_for_performance_bar_plot[organ][3])
        nmilocal_err[:, organ_count] = [nmilocal_mean[organ_count] - np.min(data_for_performance_bar_plot[organ][3]),
                                        np.max(data_for_performance_bar_plot[organ][3]) - nmilocal_mean[organ_count]]
        rmserigid_mean[organ_count] = np.mean(data_for_performance_bar_plot[organ][4])
        rmserigid_err[:, organ_count] = [rmserigid_mean[organ_count] - np.min(data_for_performance_bar_plot[organ][4]),
                                         np.max(data_for_performance_bar_plot[organ][4]) - rmserigid_mean[organ_count]]
        rmsedef_mean[organ_count] = np.mean(data_for_performance_bar_plot[organ][5])
        rmsedef_err[:, organ_count] = [rmsedef_mean[organ_count] - np.min(data_for_performance_bar_plot[organ][5]),
                                       np.max(data_for_performance_bar_plot[organ][5]) - rmsedef_mean[organ_count]]
        rmselocal_mean[organ_count] = np.mean(data_for_performance_bar_plot[organ][6])
        rmselocal_err[:, organ_count] = [rmselocal_mean[organ_count] - np.min(data_for_performance_bar_plot[organ][6]),
                                         np.max(data_for_performance_bar_plot[organ][6]) - rmselocal_mean[organ_count]]
        random_mean[organ_count] = np.mean(data_for_performance_bar_plot[organ][7])
        random_err[:, organ_count] = [random_mean[organ_count] - np.min(data_for_performance_bar_plot[organ][7]),
                                      np.max(data_for_performance_bar_plot[organ][7]) - random_mean[organ_count]]
        organ_count = organ_count + 1

    x = np.arange(len(organ_list))  # the label locations
    width = 0.1  # the width of the bars
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    fig, ax = plt.subplots()
    ax.bar(x - 3.5 * width, oracle_mean, width, yerr=oracle_err, label='Oracle')
    ax.bar(x - 2.5 * width, nmirigid_mean, width, yerr=nmirigid_err, label='NMI Affine')
    ax.bar(x - 1.5 * width, nmidef_mean, width, yerr=nmidef_err, label='NMI Def')
    ax.bar(x - 0.5 * width, nmilocal_mean, width, yerr=nmilocal_err, label='NMI Local Def')
    ax.bar(x + 0.5 * width, rmserigid_mean, width, yerr=rmserigid_err, label='RMSE Affine')
    ax.bar(x + 1.5 * width, rmsedef_mean, width, yerr=rmsedef_err, label='RMSE Def')
    ax.bar(x + 2.5 * width, rmselocal_mean, width, yerr=rmselocal_err, label='RMSE Local Def')
    ax.bar(x + 3.5 * width, random_mean, width, yerr=random_err, label='Random')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average DSC over 12 test cases')
    ax.set_title('Average DSC of 10 selected atlases')
    ax.set_xticks(x)
    ax.set_xticklabels(organ_list)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
              borderaxespad=0, frameon=True, prop={'size': 8})
    fig.tight_layout()
    plt.savefig('C:\\Mark\\Book\\Atlas selection code\\Figures\\Performance bar plot.png')

    # Plot average performance by rank for all organs
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.margins(y=0.2)
    ax = plt.gca()
    for organ in organ_list:
        plt.plot(performance_for_organ[organ], label=organ)
    plt.annotate('Random', xy=(24.5, 0.9), xytext=(27, 1.048), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('Oracle', xy=(5.5, 0.95), xytext=(0, 1.048), arrowprops=dict(facecolor='black', shrink=0.05))
    draw_brace(ax, (12, 22), 'Image-based', 0.95)
    plt.ylabel('Average DSC over 12 test cases')
    plt.xlabel('Rank by performance')
    plt.title('Ranked performance for each organ')
    plt.legend(loc='lower left')
    fig.tight_layout()
    plt.savefig('C:\\Mark\\Book\\Atlas selection code\\Figures\\Performance by organ.png')

    return


if __name__ == '__main__':
    main()
