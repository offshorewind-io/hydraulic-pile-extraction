import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from finite_element_model import solve_optimisation, normalised_hydraulic_head
import plotly.io as pio

pio.templates.default = "plotly_white"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    html.H1('Hydraulic pile extraction - finite element limit analysis'),
    html.P('This is a demonstration app, it should not be used in decision making processes without supervision from a qualified engineer. For consultancy services on software implementation or geotechnical analysis, please contact info@offshorewind.io'),

    dbc.Row(
        [
            dbc.Col(html.Div([
                html.Label('Pile Diameter (m)'),
                dbc.Input(id='pile-diameter', type='number', value=8),
                html.Label('Pile Embedded Length (m)'),
                dbc.Input(id='pile-embedded-length', type='number', value=24),
                html.Label('Pile wall thickness (mm)'),
                dbc.Input(id='pile-wall-thickness', type='number', value=100),
                html.Label('Pile weight top (t)'),
                dbc.Input(id='pile-weight-top', type='number', value=385),
                html.Label('Steel submerged unit weight (kN/m3)'),
                dbc.Input(id='steel-unit-weight', type='number', value=67.2),
                html.Label('Soil type'),
                dbc.Select(id="soil-type", value="sand",
                    options=[
                        {"label": "sand", "value": "sand"},
                        {"label": "clay", "value": "clay"},
                    ],
                ),
                html.Label('Soil submerged unit weight (kN/m3)'),
                dbc.Input(id='soil-unit-weight', type='number', value=10),
                html.Label('Initial interface friction ratio (-)'),
                dbc.Input(id='beta-0', type='number', value=0.289),
                html.Label('Pressure induced interface friction ratio (-)'),
                dbc.Input(id='beta-p', type='number', value=0.385),
                html.Label('Undrained shear strength (kPa)'),
                dbc.Input(id='su', type='number', value=100),
                html.Label('Interface factor (alpha) (-)'),
                dbc.Input(id='alpha', type='number', value=0.5),
            ]), md=3),
            dbc.Col(dcc.Graph(id='graph'), md=9),
        ])
])
@app.callback(
    dash.dependencies.Output('graph', 'figure'),
    [dash.dependencies.Input('pile-diameter', 'value'),
     dash.dependencies.Input('pile-embedded-length', 'value'),
     dash.dependencies.Input('pile-wall-thickness', 'value'),
     dash.dependencies.Input('pile-weight-top', 'value'),
     dash.dependencies.Input('soil-type', 'value'),
     dash.dependencies.Input('soil-unit-weight', 'value'),
     dash.dependencies.Input('steel-unit-weight', 'value'),
     dash.dependencies.Input('beta-0', 'value'),
     dash.dependencies.Input('beta-p', 'value'),
     dash.dependencies.Input('su', 'value'),
     dash.dependencies.Input('alpha', 'value')])
def update_figure(D, L, t_mm, W_top_t, soil_type, gamma_s_kN3, gamma_p_kN3, beta_0, beta_p, s_u_kPa, alpha):

    t = t_mm / 1000
    W_top = W_top_t * 9.81 * 1000
    n_elements = 20
    s_u = s_u_kPa * 1e3
    gamma_s = gamma_s_kN3 * 1e3
    gamma_p = gamma_p_kN3 * 1e3

    solution = solve_optimisation(L, D, t, n_elements, soil_type, W_top, beta_0, beta_p,
        gamma_p, gamma_s, alpha, s_u)

    sigma = solution['x'][:-1]
    bop = solution['x'][-1]
    z_n = np.linspace(0, L, n_elements + 1)

    k_e = [1e-4] * n_elements
    h_norm = normalised_hydraulic_head(z_n, k_e, D, t)
    h = h_norm * bop

    z_e = [0] * (2 * n_elements)
    z_e[0::2] = z_n[:-1]
    z_e[1::2] = z_n[1:]

    sigma_p1_e = sigma[0::6]
    sigma_p2_e = sigma[1::6]
    sigma_s1_e = sigma[2::6]
    sigma_s2_e = sigma[3::6]
    tau_e_e = sigma[4::6]
    tau_i_e = sigma[5::6]

    sigma_pn = [0] * (2 * n_elements)
    sigma_pn[0::2] = sigma_p1_e
    sigma_pn[1::2] = sigma_p2_e

    sigma_sn = [0] * (2 * n_elements)
    sigma_sn[0::2] = sigma_s1_e
    sigma_sn[1::2] = sigma_s2_e

    tau_en = [0] * (2 * n_elements)
    tau_en[0::2] = tau_e_e
    tau_en[1::2] = tau_e_e

    tau_in = [0] * (2 * n_elements)
    tau_in[0::2] = tau_i_e
    tau_in[1::2] = tau_i_e

    z_n = np.linspace(0, L, n_elements + 1)
    L_e = z_n[1:] - z_n[:-1]
    sigma_vn = np.insert(np.cumsum(gamma_s * L_e), 0, 0)

    fig = make_subplots(rows=1, cols=5, shared_yaxes=True, subplot_titles=(
        "Soil stresses", "Shear stresses", "Pile stress", "Water pressure", "Soil type"))

    colors = {'clay': 'steelblue',
              'sand': 'firebrick'}

    fig.add_trace(go.Scatter(x=sigma_sn, y=z_e, name='σ_i', line={'color': 'firebrick'}), 1, 1)
    fig.add_trace(go.Scatter(x=sigma_vn, y=z_n, name='σ_o', line={'color': 'steelblue'}), 1, 1)
    fig.add_trace(go.Scatter(x=tau_in, y=z_e, name='τ_i', line={'color': 'firebrick'}), 1, 2)
    fig.add_trace(go.Scatter(x=tau_en, y=z_e, name='τ_o', line={'color': 'steelblue'}), 1, 2)
    fig.add_trace(go.Scatter(x=sigma_pn, y=z_e, name='σ_p', line={'color': 'black'}), 1, 3)
    fig.add_trace(go.Scatter(x=h, y=z_n, name='p', line={'color': 'steelblue'}), 1, 4)
    fig.add_trace(go.Bar(x=[""], y=[L], name=soil_type, marker={'color': colors[soil_type]}), 1, 5)

    fig['layout']['yaxis']['title'] = 'Depth (m)'
    fig['layout']['xaxis']['title'] = 'σ (Pa)'
    fig['layout']['xaxis2']['title'] = 'τ (Pa)'
    fig['layout']['xaxis3']['title'] = 'σ (Pa)'
    fig['layout']['xaxis4']['title'] = 'p (Pa)'

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=600, title_text="Break out pressure: " + str(bop) + " Pa")

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
