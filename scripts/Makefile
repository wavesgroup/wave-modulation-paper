.PHONY: all figures

all: figures

figures: hydrodynamic_modulation_diagram \
         analytical_solutions_with_phase \
         numerical_vs_analytical_solutions \
         effective_gravity \
         inhomogeneity_tendencies \
         numerical_simulations \
         numerical_simulations_with_phase \
         numerical_simulations_ramp_properties \
         modulation_contributors \
	 velocity_error \
	 action_conservation_criteria

hydrodynamic_modulation_diagram: plot_modulation_diagram.py
	python3 plot_modulation_diagram.py

analytical_solutions_with_phase: plot_analytical_solutions_with_phase.py
	python3 plot_analytical_solutions_with_phase.py

effective_gravity: plot_effective_gravity.py
	python3 plot_effective_gravity.py

modulation_contributors: plot_modulation_contributors.py
	python3 plot_modulation_contributors.py

inhomogeneity_tendencies: plot_inhomogeneity_tendencies.py
	python3 plot_inhomogeneity_tendencies.py

numerical_vs_analytical_solutions: plot_numerical_vs_analytical_solutions.py
	python3 plot_numerical_vs_analytical_solutions.py

numerical_simulations_with_phase: plot_numerical_vs_analytical_solutions_with_phase.py
	python3 plot_numerical_vs_analytical_solutions_with_phase.py

numerical_simulations: plot_numerical_simulations.py
	python3 plot_numerical_simulations.py

numerical_simulations_ramp_properties: plot_numerical_simulations_ramp_properties.py
	python3 plot_numerical_simulations_ramp_properties.py

velocity_error: plot_velocity_error.py
	python3 plot_velocity_error.py

action_conservation_criteria: plot_action_conservation_criteria.py
	python3 plot_action_conservation_criteria.py
	python3 plot_velocity_error.py
