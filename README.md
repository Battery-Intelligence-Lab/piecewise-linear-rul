## README for code release

The code in this repository is the code behind results in published work:
"*Piecewise-linear modelling with automated feature selection for Li-ion battery end-of-life prognosis*". 
Questions regarding the code should be sent to either or both of:
    samuel.greenbank.battery@gmail.com
    david.howey@eng.ox.ac.uk

The code is owned by Samuel Greenbank and David Howey, University of Oxford, 2021. Any use of the code should cite the relevant paper or thesis.

### Code intention
We want to be clear, this code is not a piece of software ready for production in a real world setting. Instead, the code should allow a user to run the model in the paper.

Second, the code quality does not represent an up to date representation of what we are capable of. We have left the code in its original condition for the purposes of honesty. We apologise for any ambiguities in variable names or structure that appear as a result. Please do ask if unsure.

We cannot accept edits to the code, precisely because it presents the work of the paper. However we would be very interested in hearing about potential improvements through the above emails. We do encourage you to play with the code, after all the original version will be safely stored here.

### Using the code
If you have data in the correct form then the code will run if you just press play on the *soh_prognosis_trial*. The correct form is a .csv file containing columns for health, time and input features.

Files:
- *feature_engineering.py* : feature selection and preparing data for the regression tools
- *piecewise_linear_regression.py* : a purely regression based tool. Acts on its own.
- *results.py* : presents results
- *soh_prognosis_plr.py* : code behind a battery health regression tool
- *soh_prognosis_trial.py* : trial running a number of repeats on a training and test set.

### Missing work
Not all plots in the paper can be directly reproduced from this code. You would have to set up your own trials for comparison or sentivity analysis. We opted for this reduced form for simplicity.

### Required citations
The work developing this code would not have been possible without the funding of Siemens PLC and EPSRC.

Further, the data behind the work is from the following two papers:
- K.A. Severson, P.M. Attia, N. Jin, N. Perkins, B. Jiang, Z. Yang, M.H. Chen, M. Aykol, P.K. Herring, D. Fraggedakis, M.Z. Bazant, S.J. Harris, W.C. Chueh, and R.D. Braatz., "_Data-driven prediction of battery cycle life before capacity degradation_," Nature Energy, **4**:383–391, 2019.
- P.M. Attia, A. Grover, N. Jin, K.A. Severson, T.M. Markov, Y.-H. Liao, M.H. Chen, B. Cheong, N. Perkins, Z. Yang, P.K. Herring, M. Aykol, S.J. Harris, R.D. Braatz, S. Ermon, and W.C. Chueh., "_Closed-loop optimization of fast-charging protocols for batteries with machine learning_," Nature, **578**:397–402, 2020.