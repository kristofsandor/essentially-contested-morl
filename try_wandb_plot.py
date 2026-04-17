import wandb
wandb.init(project="MORL-Baselines")

# generate integers until 20
data = np.arange(20).reshape(-1, 4)

columns = ["blue_triangle", "blue_circle", "red_triangle", "red_circle"]
pf_table = wandb.Table(data=data, columns=columns)

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        wandb.log({f"pf_{columns[i]}_vs_{columns[j]}": wandb.plot_table(
            vega_spec_name="kristofs-ai/MORL-Baselines/basic_scatter_plot",
            data_table=pf_table,
            fields={"x": f"{columns[i]}", "y": f"{columns[j]}"},
        )})

wandb.finish()