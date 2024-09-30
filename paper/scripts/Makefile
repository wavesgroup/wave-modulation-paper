.PHONY: all figures

all: figures

figures: analytical_solutions_with_phase \
         numerical_vs_analytical_solutions \
    	 effective_gravities \
	     inhomogeneity_tendencies \
         numerical_simulations \
         numerical_simulations_with_phase \
	     numerical_simulations_ramp_properties

analytical_solutions_with_phase: plot_analytical_solutions_with_phase.py
	python3 plot_analytical_solutions_with_phase.py

effective_gravities: compare_effective_gravities.py
	python3 compare_effective_gravities.py

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
