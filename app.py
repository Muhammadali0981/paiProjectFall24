from flask import Flask, render_template
from analysis.Graphs import generate_graphs
from analysis.KNeighboursRegressor import perform_knn_regression
from analysis.statistical_analysis import perform_statistical_analysis

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/graphs')
def graphs():
    graph_results = generate_graphs()
    return render_template('graphs.html', 
                           box_plot=graph_results['box_plot'], 
                           distribution_plots=graph_results['distribution_plots'], 
                           correlation_plot=graph_results['correlation_plot'], 
                           regression_plot=graph_results['regression_plot'])

@app.route('/regression')
def regression():
    regression_results = perform_knn_regression()
    return render_template('regression.html', 
                           y1_best_k=regression_results['Y1']['best_k'],
                           y1_mse=regression_results['Y1']['mse'],
                           y1_r2=regression_results['Y1']['r2'],
                           y2_best_k=regression_results['Y2']['best_k'],
                           y2_mse=regression_results['Y2']['mse'],
                           y2_r2=regression_results['Y2']['r2'],
                           regression_plot=regression_results['regression_plot'])

@app.route('/statistics')
def statistics():
    statistical_results = perform_statistical_analysis()
    return render_template('statistics.html', 
                           statistical_groups=statistical_results['groups'],
                           statistical_plot=statistical_results['plot'])

if __name__ == '__main__':
    app.run(debug=True)