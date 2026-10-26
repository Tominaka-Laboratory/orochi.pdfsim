def explore_subgroup_pathways(self, top_k=5, output="text", **kwargs):
    """
    see simulator
    """
    from orochi.fit.pdf.least_squares import Optimizer as LeastSquaresOptimizer
    from orochi.fit.pdf.bayes import Optimizer as BayesOptimizer
    import pandas as pd

    if self.obs_x is None or self.obs_y is None:
        self._logger.error("No obs data were set in simulator")
        return {}

    self.phase.make_higher_symmetry_structure()

    ## Main optimizer
    default_config_optimizer = {"type": "bayes", "fit_min": 1.5, "fit_max": 15.0, 
                                "n_trials": 100, "n_jobs":1, "continue_study":True,
                                "score_type": "raw"}

    optimizer = kwargs.get("optimizer")
    if optimizer is None:
        opt_config = default_config_optimizer
    else:
        opt_config = {**default_config_optimizer, **optimizer}

    opt_type = opt_config.pop("type", "bayes")

    if opt_type == "lls":
        self.optimizer = LeastSquaresOptimizer()
    elif opt_type in ["bayes", "ba", "optuna"]:
        self.optimizer = BayesOptimizer()
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    self.optimizer.study_name = opt_config.pop("study_name", "hall_number")

    for key, val in opt_config.items():
        self.optimizer[key] = val

    self.optimizer.simulators = [self]
    self.optimizer.work_dir = self.work_dir
    self.optimizer.set_obs_data(self.obs_x, self.obs_y)


    ## Nest Optimizer
    default_config_optimizer2 = {"type": "bayes", "fit_min": 1.5, "fit_max": 15.0, 
                                 "n_trials": 50, "n_jobs":1, "continue_study":True,
                                 "score_type":"adj"}
    nest_optimizer = kwargs.get("nest_optimizer")
    if nest_optimizer is None:
        opt_config = default_config_optimizer2
    else:
        opt_config = {**default_config_optimizer2, **nest_optimizer}

    opt_type = opt_config.pop("type", "bayes")

    if opt_type == "lls":
        self.nest_optimizer = LeastSquaresOptimizer()
    elif opt_type in ["bayes", "ba", "optuna"]:
        self.nest_optimizer = BayesOptimizer()
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    nest_study_name = opt_config.pop("study_name", "sub_trial")

    for key, val in opt_config.items():
        self.nest_optimizer[key] = val

    self.nest_optimizer.simulators = [self]
    self.nest_optimizer.work_dir = self.work_dir
    self.nest_optimizer.set_obs_data(self.obs_x, self.obs_y)

    ## refinements

    from .refinement import _get_all_refinement_params
    conditions = _get_all_refinement_params(self, **kwargs)
    self.optimizer.score_function = self.nest_optimizer.use_as_score_function(conditions, nest_study_name)

    ##run
    from .refinement import _get_hall_number_refinement_params
    hall_conditions = _get_hall_number_refinement_params(self)
    self.optimizer.run_with_list_conditions(conditions=hall_conditions)

    ##update with the hall number
    best_params = self.optimizer.study.best_params     
    curr_hall = self.phase["hall_number"]
    visited_path = [str(curr_hall)]

    while True:
        self.phase()
        path_str = "_".join(visited_path)
        hall_key = f"hall_path_{path_str}"
        if hall_key not in best_params:
            break
        selected_hall = str(best_params[hall_key])
        if selected_hall == "stay":
            break
        trans_key = f"trans_path_{path_str}_to_{selected_hall}"
        selected_trans = str(best_params.get(trans_key, "a,b,c"))

        self.phase.make_subgroup_structure(selected_hall, selected_trans)

        clean_trans = selected_trans.replace(",", "").replace(";", "_").replace("/", "").replace(" ", "")
        visited_path.append(f"{selected_hall}({clean_trans})")
        curr_hall = selected_hall

    ##update with the nested optimized parameters
    import optuna
    best_trial = self.optimizer.study.best_trial
    nest_study_name = f"{nest_study_name}_{best_trial.number}"

    self.nest_optimizer.study = optuna.load_study(
        study_name=nest_study_name,
        storage=self.nest_optimizer.storage
    )
    self.nest_optimizer.call_back_best_params()

    best_scores = self.optimizer.study.best_value
    best_params = self.optimizer.study.best_params

    best_scores_nest = self.nest_optimizer.study.best_value
    best_params_nest = self.nest_optimizer.study.best_params

    study = self.optimizer.study
    summary_df, llm_input_text = summarize_optuna_trials_for_llm(study, top_k=top_k)

    if output == "text":
        return llm_input_text
    else:               
        return {"best_scores": best_scores, 
                "best_params": best_params,
                "nest_scores": best_scores_nest,
                "nest_params": best_params_nest,
                "summary_df": summary_df
                }




def summarize_optuna_trials_for_llm(study, top_k=10):
    """
    Consolidate Optuna trial history into a clean, human/LLM-readable chronological pathway summary.

    Args:
        study (optuna.Study): The completed Optuna study object from space group exploration.
        top_k (int): Number of top candidate pathways to include in the output.

    Returns:
        tuple: (summary_df, llm_prompt_text)
            - summary_df (pd.DataFrame): Grouped DataFrame sorted by best RMSE.
            - llm_prompt_text (str): Formatted English text report.
    """
    df = study.trials_dataframe()
    df_complete = df[df['state'] == 'COMPLETE'].copy()
    if df_complete.empty:
        return None, "No completed trials found."

    def extract_full_chronological_path(row):
        hall_cols = [c for c in row.index if c.startswith('params_hall_path_') and pd.notna(row[c])]
        if not hall_cols:
            return "Initial Structure"

        root_col = min(hall_cols, key=len)
        initial_hall = root_col.replace('params_hall_path_', '')

        curr_hall = initial_hall
        visited_path = [initial_hall]
        path_elements = [initial_hall]

        while True:
            path_str = "_".join(visited_path)
            hall_key = f"params_hall_path_{path_str}"

            if hall_key not in row or pd.isna(row[hall_key]):
                break

            selected_hall = str(row[hall_key])
            if selected_hall == "stay":
                break

            trans_key = f"params_trans_path_{path_str}_to_{selected_hall}"
            selected_trans = str(row[trans_key]) if (trans_key in row and pd.notna(row[trans_key])) else "unknown"

            path_elements.append(f"--[{selected_trans}]--> {selected_hall}")

            clean_trans = selected_trans.replace(",", "").replace(";", "_").replace("/", "").replace(" ", "")
            visited_path.append(f"{selected_hall}({clean_trans})")
            curr_hall = selected_hall

        return " ".join(path_elements)

    df_complete['path_summary'] = df_complete.apply(extract_full_chronological_path, axis=1)

    summary_df = df_complete.groupby('path_summary')['value'].agg(
        min_rmse='min',
        mean_rmse='mean',
        std_rmse='std',
        count='count'
    ).reset_index()

    summary_df['std_rmse'] = summary_df['std_rmse'].fillna(0.0)
    summary_df = summary_df.sort_values(by='min_rmse', ascending=True).reset_index(drop=True)

    top_df = summary_df.head(top_k)

    llm_prompt_text = f"[Optuna Subgroup Search Summary (Top {len(top_df)})]\n"
    llm_prompt_text += "Note: Duplicated parameter paths are consolidated.\n\n"

    for idx, row in top_df.iterrows():
        llm_prompt_text += f"Rank {idx + 1}:\n"
        llm_prompt_text += f"  - Transformation Path: {row['path_summary']}\n"
        llm_prompt_text += f"  - Best RMSE: {row['min_rmse']:.4f}\n"
        llm_prompt_text += f"  - Mean RMSE: {row['mean_rmse']:.4f} ± {row['std_rmse']:.4f}\n"
        llm_prompt_text += f"  - Count: {int(row['count'])}\n\n"

    return summary_df, llm_prompt_text