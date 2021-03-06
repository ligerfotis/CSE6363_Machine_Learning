import matplotlib.pyplot as plt


def plot_data(dataset, sim_data_0, sim_data_1, labels):
    title = "Real_vs_Simulated_Data_Men"
    plt.figure(title)
    plt.title("Real vs Simulated Data Men Class")
    plt.xlabel("heights")
    plt.ylabel("weights")
    plt.scatter(sim_data_0[:, 0], sim_data_0[:, 1], color='blue', label="Simulated Data")
    plt.scatter(dataset[labels == 'M'][:, 0], dataset[labels == 'M'][:, 1], color='r', label="Training Data")
    plt.legend()
    plt.savefig(title)

    title = "Real_vs_Simulated_Data_Women"
    plt.figure(title)
    plt.title("Real vs Simulated Data Women Class")
    plt.xlabel("heights")
    plt.ylabel("weights")
    plt.scatter(sim_data_1[:, 0], sim_data_1[:, 1], color='blue', label="Simulated Data")
    plt.scatter(dataset[labels == 'W'][:, 0], dataset[labels == 'W'][:, 1], color='r', label="Training Data")
    plt.legend()
    plt.savefig(title)

    title = "Real_vs_Simulated_Data_All_Classes"
    plt.figure(title)
    plt.title("Real vs Simulated Data Women")
    plt.xlabel("heights")
    plt.ylabel("weights")
    plt.scatter(sim_data_0[:, 0], sim_data_0[:, 1], color='green', label="Simulated Data Men")
    plt.scatter(dataset[labels == 'M'][:, 0], dataset[labels == 'M'][:, 1], color='orange', label="Training Data Men")
    plt.scatter(sim_data_1[:, 0], sim_data_1[:, 1], color='blue', label="Simulated Data Women")
    plt.scatter(dataset[labels == 'W'][:, 0], dataset[labels == 'W'][:, 1], color='r', label="Training Data Women")
    plt.legend()
    plt.savefig(title)
