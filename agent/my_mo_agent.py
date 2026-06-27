from typing import Optional, Literal

from morl_baselines.common.morl_algorithm import MOAgent

class MyMOAgent(MOAgent):

    
    def setup_wandb(
        self,
        project_name: str,
        experiment_name: str,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        mode: Literal["online", "offline", "disabled", "shared"] | None = "online",
    ) -> None:
        """Initializes the wandb writer.

        Args:
            project_name: name of the wandb project. Usually MORL-Baselines.
            experiment_name: name of the wandb experiment. Usually the algorithm name.
            entity: wandb entity. Usually your username but useful for reporting other places such as openrlbenmark.

        Returns:
            None
        """
        self.experiment_name = experiment_name
        self.full_experiment_name = experiment_name
        import wandb

        config = self.get_config()
        config["algo"] = self.__class__.__name__
        # looks for whether we're using a Gymnasium based env in env_variable

        wandb.init(
            project=project_name,
            entity=entity,
            config=config,
            name=self.experiment_name,
            monitor_gym=False,
            save_code=True,
            group=group,
            mode=mode,
        )
        # The default "step" of wandb is not the actual time step (gloabl_step) of the MDP
        wandb.define_metric("*", step_metric="global_step")