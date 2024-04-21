from flask import Flask, render_template, request
import io
import base64
import openturns as ot
import numpy as np
from matplotlib import pylab as plt
import openturns.viewer as viewer

app = Flask(__name__)

def create_ot_function(func, problem):
    inputDimension = problem['num_vars']
    ot_func = ot.PythonFunction(inputDimension, 1, func)
    return ot_func

def create_input_distribution(problem):
    distributions = []
    for i in range(problem['num_vars']):
        bounds = problem['bounds'][i]
        distribution_type = problem['distributions'][i].lower()

        if distribution_type == 'uniform':
            distributions.append(ot.Uniform(bounds[0], bounds[1]))
        elif distribution_type == 'normal':
            mean = (bounds[0] + bounds[1]) / 2
            std_dev = (bounds[1] - bounds[0]) / 4  # Assuming 95% of values lie within 4 standard deviations
            distributions.append(ot.Normal(mean, std_dev))
        elif distribution_type == 'lognormal':
            # Convert bounds to parameters for the lognormal distribution
            # Here, we make a simplifying assumption about the shape of the distribution
            scale = np.sqrt(bounds[0] * bounds[1])  # Geometric mean
            shape = (np.log(bounds[1]) - np.log(bounds[0])) / 6  # Roughly capturing the range
            distributions.append(ot.LogNormal(np.log(scale), shape, 0.0))
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")

    distribution = ot.ComposedDistribution(distributions)
    distribution.setDescription(problem['names'])
    return distribution

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        function_str = request.form['function']
        num_vars = int(request.form['num_vars'])
        names = [name.strip() for name in request.form['names'].split(',')]
        bounds = []
        for bound_str in request.form['bounds'].split('\n'):
            if bound_str.strip():
                bound = [eval(b.strip()) for b in bound_str.split(',')]
                bounds.append(bound)
        distributions = [dist.strip() for dist in request.form['distributions'].split(',')]

        user_code = f"""
def function_of_interest(x):
    import numpy as np
    {', '.join(names)} = x
    return [{function_str}]

problem = {{
    'num_vars': {num_vars},
    'names': {names},
    'bounds': {bounds},
    'distributions': {distributions}
}}
        """

        local_variables = {'np': np, 'ot': ot}


        try:
            exec(user_code, {}, local_variables)
        except Exception as e:
            return render_template('index.html', error=f"Error in execution: {e}")

        function_of_interest = local_variables.get('function_of_interest')
        problem = local_variables.get('problem')

        if not function_of_interest or not problem:
            return render_template('index.html', error="Function or problem definition not provided correctly.")

        # Proceed with the provided OpenTURNS workflow
        ot.Log.Show(ot.Log.NONE)
        ot_function = create_ot_function(function_of_interest, problem)
        input_distribution = create_input_distribution(problem)

        # Setup the analysis for ExpectationSimulationAlgorithm
        inputVector = ot.RandomVector(input_distribution)
        outputVector = ot.CompositeRandomVector(ot_function, inputVector)
        algo = ot.ExpectationSimulationAlgorithm(outputVector)
        algo.setMaximumOuterSampling(80000)
        algo.setBlockSize(1)
        algo.setCoefficientOfVariationCriterionType("NONE")
        algo.run()
        result = algo.getResult()

        expectation = result.getExpectationEstimate()
        expectationVariance = result.getVarianceEstimate()
        standardDeviation = result.getStandardDeviation()
        expectationDistribution = result.getExpectationDistribution()



        # Generate samples and evaluate the model
        n_samples = 10000
        sampleX = input_distribution.getSample(n_samples)
        sampleY = ot_function(sampleX)

        # Adjust the figure layout
        fig = plt.figure(figsize=(15, 15))  # Adjusted figure size
        gs = fig.add_gridspec(4, problem['num_vars'], height_ratios=[2, 2, 1, 1])  # Adjusted grid layout

        # New: Add the convergence plot to the upper left
        ax_convergence = fig.add_subplot(gs[0, :problem['num_vars']//2])
        convergence_graph = algo.drawExpectationConvergence()
        _ = viewer.View(convergence_graph, figure=fig, axes=[ax_convergence])
        ax_convergence.set_title("Convergence Plot")
        plt.gca().annotate(
            f"Estimated mean: {expectation[0]:.5f}\nStandard deviation: {standardDeviation[0]:.5f}", 
            xy=(0.5, 0.1), 
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="black", lw=1)
        )

        # New: Add the PDF of the asymptotic normal distribution to the upper right
        ax_distribution = fig.add_subplot(gs[0, problem['num_vars']//2:])
        distribution_graph = expectationDistribution.drawPDF()
        distribution_graph.setTitle("Normal asymptotic distribution of the mean estimate")
        _ = viewer.View(distribution_graph, figure=fig, axes=[ax_distribution])
        ax_distribution.set_title("Normal Distribution of Mean Estimate")
        plt.gca().annotate(
            f"Normal($\mu$ = {expectation[0]:.5f}, $\sigma$ = {standardDeviation[0]:.5f})", 
            xy=(0.5, 0.7), 
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="black", lw=1)
        )

        # Plot the X vs Y graphs in the second row, spanning all columns
        for i in range(problem['num_vars']):
            ax = fig.add_subplot(gs[1, i])
            graph = ot.Graph("", problem['names'][i], "Y", True, "")
            cloud = ot.Cloud(sampleX[:, i], sampleY)
            graph.add(cloud)
            _ = viewer.View(graph, figure=fig, axes=[ax])
            ax.set_title(f"{problem['names'][i]} vs Y")

        ax_histogram = fig.add_subplot(gs[2, :problem['num_vars']//2])
        histogram_graph = ot.HistogramFactory().build(sampleY).drawPDF()
        _ = viewer.View(histogram_graph, figure=fig, axes=[ax_histogram])
        ax_histogram.set_title("Histogram of Y")

        ax_sensitivity = fig.add_subplot(gs[2, problem['num_vars']//2:])
        sie = ot.SobolIndicesExperiment(input_distribution, 1000, True)
        inputDesign = sie.generate()
        inputDesign.setDescription(problem['names'])
        outputDesign = ot_function(inputDesign)

        # sensitivityAnalysis = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, 1000)


        sensitivity_method = request.form.get('sensitivity_method', 'SaltelliSensitivityAlgorithm')
        
        # Based on the selected method, use the corresponding OpenTURNS algorithm
        if sensitivity_method == 'MartinezSensitivityAlgorithm':
            sensitivityAnalysis = ot.MartinezSensitivityAlgorithm(inputDesign, outputDesign, 1000)
        elif sensitivity_method == 'JansenSensitivityAlgorithm':
            sensitivityAnalysis = ot.JansenSensitivityAlgorithm(inputDesign, outputDesign, 1000)
        elif sensitivity_method == 'MauntzKucherenkoSensitivityAlgorithm':
            sensitivityAnalysis = ot.MauntzKucherenkoSensitivityAlgorithm(inputDesign, outputDesign, 1000)
        else:  # Default to SaltelliSensitivityAlgorithm
            sensitivityAnalysis = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, 1000)
        

        sensitivity_graph = sensitivityAnalysis.draw()

        viewer.View(sensitivity_graph, figure=fig, axes=[ax_sensitivity])

        plt.tight_layout()

        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    #     return render_template('index.html', plot_url=plot_url)
    # return render_template('index.html')

        # return render_template('index.html', plot_url=plot_url, 
        #                        function_str=function_str, num_vars=str(num_vars), 
        #                        names=",".join(names), bounds="\n".join([",".join(map(str, b)) for b in bounds]), 
        #                        distributions=",".join(distributions))
    
        return render_template('index.html', plot_url=plot_url, 
                               function_str=function_str, num_vars=str(num_vars), 
                               names=",".join(names), bounds="\n".join([",".join(map(str, b)) for b in bounds]), 
                               distributions=",".join(distributions),
                               sensitivity_method=sensitivity_method)



    # This line should be added at the end of your index() function, replacing the existing GET return
    # return render_template('index.html', function_str='', num_vars='', names='', bounds='', distributions='')
    return render_template('index.html', function_str='', num_vars='', names='', bounds='', distributions='', sensitivity_method='SaltelliSensitivityAlgorithm')




if __name__ == '__main__':
    app.run(debug=True)
