import plotly.offline as offline
import plotly.graph_objs as go
from math import log10,log2

def plot_bubble_chart():
    '''
    Function used to generate the bubble chart in the project report.
    Hard-coded lists are representative of the models listed in Table 1.
    '''
    ps = [.89, .85, .65, .63, .60, .58, .54, .51, .47, .44, .40, .39, .34, .31]
    colors = [ 'grey', 'blue', 'red', 'red', 'green', 'green', 'green', 'green', 'red', 'green', 'green', 'green', 'green', 'green']
    precision = [ 100*p for p in ps ]
    numDocs=[799, 799, 799, 799, 250, 200, 150, 100, 799, 50, 50, 25, 100, 10]
    numIters=[1000, 1, 1000, 1000, 150, 150, 100, 250, 1000, 250, 50, 100, 25, 250]
    trainingTimes=[3410,.0459,48600,3410,6120,2920,2930,1110,3410,660,285,352,368,381]
    logd_times = [ 3**abs(log10(t)) for t in trainingTimes]
    trace0 = go.Scatter(
        x = numDocs,
        y = precision,
        mode='markers+text',
        text = ['Madsen DCM GD', 'MultinomialNB', 'DCM GD', 'DCM GD (Minka, 0.85)', 'DCM (150,250)','DCM(150,200)','DCM(100,150)','DCM(250,100)','DCM GD (Minka, 0.01)', 'DCM(250,50)', 'DCM(50,50)','DCM(100,25)','DCM(25,100)','DCM(250,10)'],
        textposition=['middle center', 'middle left', 'middle center', 'middle left', 'middle center', 'middle center', 'middle center', 'middle center', 'middle left', 'middle center','middle center','middle center','middle center', 'middle center'],
        #textposition='middle center',
        marker=dict(
            size=logd_times,
            sizemode='diameter',
            sizeref=1,
            color=colors,
            sizemin=5
        )
    )

    layout = dict(
        title = "Plateauing R.O.I. between Number of Training Documents and Precision",
        xaxis = dict(
            title = "Number of Documents per Training Iteration"
        ),
        yaxis = dict(
            title = "Precision (%)"
        ),
        annotations=[
        dict(
            x=300,
            y=100,
            xref='x',
            yref='y',
            text='Blue: Multinomial Classifier',
            font=dict(
                size=16,
                color='blue'
            ),
            showarrow=False
        ),
        dict(
            x=300,
            y=95,
            xref='x',
            yref='y',
            text = 'Red: DCM Classifier trained using GD',
            font=dict(
                size=16,
                color='red'
            ),
            showarrow=False
        ),
        dict(
            x=300,
            y=90,
            xref='x',
            yref='y',
            text='Green: DCM Classifier trained using minibatch GD',
            font=dict(
                size=16,
                color='green'
            ),
            showarrow=False
        ),
        dict(
            x=300,
            y=85,
            xref='x',
            yref='y',
            text='Gray: DCM Classifier trained using GD from Madsen et al.',
            font=dict(
                size=16,
                color='gray'
            ),
            showarrow=False
        )
        ]
    )
    

    fig = go.Figure(data=[trace0], layout=layout)
    offline.plot(fig, filename='plateauing-r.o.i.-between-number-of-training-documents-and-precision.html')

if __name__ == '__main__':
    plot_bubble_chart()