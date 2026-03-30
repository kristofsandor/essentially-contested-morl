import wandb
import numpy as np
import matplotlib.pyplot as plt


def visualize_front_general(pareto_front, columns):
    pf_arr = np.array((list(pareto_front)))

    # 1) Build a W&B table
    pf_table = wandb.Table(columns=columns, data=pf_arr)

    # 2) Log 2D Pareto front projections as scatter plots
    # go through all combinations of columns
    n_pairs = len(columns) * (len(columns) - 1) // 2
    fig, axs = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4))
    if n_pairs == 1: # subscriptable even if there's only 1 element
        axs = [axs]

    pair_count = 0
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            wandb.log(
                {
                    f"pf_{columns[i]}_vs_{columns[j]}": wandb.plot.scatter(
                        pf_table,
                        x=columns[i],
                        y=columns[j],
                        title=f"Pareto front: {columns[i]} vs {columns[j]}",
                    )
                }
            )
            ax = axs[pair_count]
            ax.scatter(pf_arr[:, i], pf_arr[:, j], color="black")
            ax.set_xlabel(columns[i])
            ax.set_ylabel(columns[j])
            ax.set_title(f"Pareto front: {columns[i]} vs {columns[j]}")
                        
            pair_count += 1

    plt.show()
    plt.close(fig)
    return fig

def visualize_front(pareto_front):
    # pareto_front is currently a set of (d, g, di)
    pf_list = sorted(list(pareto_front))  # optional sort, just for consistency

    # 1) Build a W&B table
    pf_table = wandb.Table(columns=["death_penalty", "gold", "diamond"])
    for d, g, di in pf_list:
        pf_table.add_data(d, g, di)

    # 2) Log 2D Pareto front projections as scatter plots
    wandb.log({
        # color according to 3rd objective
        # Death penalty vs Gold
        "pareto_front_death_vs_gold": wandb.plot.scatter(
            pf_table,
            x="death_penalty",
            y="gold",
            title="Pareto front: death penalty vs gold",
        ),
        # Death penalty vs Diamond
        "pareto_front_death_vs_diamond": wandb.plot.scatter(
            pf_table,
            x="death_penalty",
            y="diamond",
            title="Pareto front: death penalty vs diamond",
        ),
        # Gold vs Diamond (often the most interesting trade-off)
        "pareto_front_gold_vs_diamond": wandb.plot.scatter(
            pf_table,
            x="gold",
            y="diamond",
            title="Pareto front: gold vs diamond",
        ),

    })

    pf_arr = np.array(pf_list)
    fig, axs = plt.subplots(1, 3)
    fig.set_figwidth(18)
    fig.set_figheight(5)
    axs[0].scatter(pf_arr[:, 0], pf_arr[:, 1])
    axs[1].scatter(pf_arr[:, 0], pf_arr[:, 2])
    axs[2].scatter(pf_arr[:, 1], pf_arr[:, 2])

    # axs[0].set_title(f"Pareto front for gamma")

    axs[0].set_xlabel("death_penalty")
    axs[1].set_xlabel("death_penalty")
    axs[2].set_xlabel("gold")

    axs[0].set_ylabel("gold")
    axs[1].set_ylabel("diamond")
    axs[2].set_ylabel("diamond")

    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pf_arr[:, 0], pf_arr[:, 1], pf_arr[:, 2])
    ax.set_xlabel("death_penalty")
    ax.set_ylabel("gold")
    ax.set_zlabel("diamond")
    ax.set_title("Pareto front")

    wandb.log({"pareto_front_3d": wandb.Image(fig)})

    plt.close(fig)