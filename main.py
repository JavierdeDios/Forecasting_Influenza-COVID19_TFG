import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statistics import mean

import pmdarima as pm
from pmdarima.arima.utils import ndiffs
from pmdarima.arima import auto_arima, ADFTest

from sklearn.metrics import mean_squared_error
from math import sqrt

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

register_matplotlib_converters()

plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})


def initialize(name):
    df = pd.read_csv(name)
    df = df.drop(['REGION_TYPE', 'WEIGHTED_ILI', 'UNWEIGHTED_ILI', 'AGE_0-4', 'AGE_25-49', 'AGE_25-64', 'AGE_5-24',
                  'AGE_50-64', 'AGE_65', 'NUM_OF_PROVIDERS'], 1)
    df = df[df.YEAR != 2009]
    df = df[df.YEAR != 2019]

    cols_to_clean = ['REGION', 'YEAR', 'WEEK', 'ILITOTAL', 'TOTAL_PATIENTS']
    for col in cols_to_clean:
        df.drop(df[df[col] == 'X'].index, inplace=True)

    df['TOTAL_PATIENTS'] = df['TOTAL_PATIENTS'].astype(int)
    df['ILITOTAL'] = df['ILITOTAL'].astype(int)
    df["WEEK_YEAR"] = df.WEEK.astype(str).str.cat(df.YEAR.astype(str), sep="-")
    df['WEEK_YEAR'] = pd.to_datetime(df['WEEK_YEAR'] + '-1', format='%W-%Y-%w')

    eps = 10e-9
    df["ILI_RATIO"] = df["ILITOTAL"] / (df["TOTAL_PATIENTS"] + eps)
    print('## Dataframe after initialize: ')
    df.info(), print()

    return df


def get_stationarity(data, state):
    # rolling statistics
    rolling_mean = data.rolling(window=10).mean()
    rolling_std = data.rolling(window=10).std()
    plt.plot(data, color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title(state + ' ILI ratio rolling Mean & standard Deviation')
    # plt.savefig('plots/' + STATE + '_stationarity.png')
    plt.show(block=False)

    # Dickey–Fuller test:
    result = adfuller(data['ILI_RATIO'], autolag='AIC')
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
    if result[1] < 0.05:
        print('RESULT: This series is stationary.')
    else:
        print('RESULT: This series is non-stationary.')


def get_ac_pac(data):
    plot_pacf(data, lags=100, method='ywm', auto_ylims=1)  # Partial autocorrelation
    # plt.title(' Partial Autocorrelation')
    plt.show()

    plot_acf(data, lags=100, auto_ylims=1)  # Autocorrelation
    # plt.title(' Autocorrelation')
    plt.show()


def autotests(data):
    adf_test = ADFTest(alpha=0.05)
    p_val, should_diff = adf_test.should_diff(data)
    print('p_value: ', p_val, ', need diff? ', should_diff)

    n_adf = ndiffs(data, test='adf')
    print('ADF: ', n_adf)

    n_kpss = ndiffs(data, test='kpss')
    print('KPPS: ', n_kpss)

    n_pp = ndiffs(data, test='pp')
    print('PP: ', n_pp)


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0, 1]   # corr
    mins = np.amin(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:, None],
                              actual[:, None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    print('mape:', mape, 'me:', me, 'mae:', mae,
          'mpe:', mpe, 'rmse:', rmse,
          'corr:', corr, 'minmax:', minmax)
    return rmse, corr


done_list = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'Delaware', 'Georgia', 'Idaho', 'Kansas', 'Kentucky',
             'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Nebraska', 'Nevada',
             'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
             'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas',
             'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin']

states_abv = ['AL', 'AK', 'AZ', 'AR', 'DE', 'GA', 'ID', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'NE', 'NV',
              'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA',
              'WA', 'WV', 'WI']

st_df = initialize('ILINetSTATES.csv')

STATE = 'Maine'

print('##################################################################################################'
      , STATE,
      '##################################################################################################')
print()

st_df_copy = st_df[st_df.REGION == STATE]  # state select
print('## Dataframe after region selection (st_df_copy): ')
st_df_copy.info()

plt.plot(st_df_copy.WEEK_YEAR, st_df_copy.TOTAL_PATIENTS)  # Total patients plot
plt.title('Total ' + STATE + ' patients from 2010 to 2018')
plt.ylabel('Number of patients')
# plt.savefig('plots/' + STATE + '_totalP.png')
plt.show()
print()

plt.plot(st_df_copy.WEEK_YEAR, st_df_copy.ILITOTAL)  # Total ILI plot
plt.title('Total ' + STATE + ' ILI from 2010 to 2018')
plt.ylabel('Number of ILI')
# plt.savefig('plots/' + STATE + '_totalILI.png')
plt.show()

st_series = st_df_copy[['WEEK_YEAR', 'ILI_RATIO']].copy()  # interesting columns to plot
st_series.set_index('WEEK_YEAR', inplace=True)
print('## Series in use (st_series): ')
print(st_series)
print()

plt.plot(st_series)
plt.title(STATE + ' ILI ratio from 2010 to 2018')  # Total ILI ratio plot
plt.ylabel('ILI ratio')
# plt.savefig('plots/' + STATE + '_ILIRATIO.png')
plt.show()

print('## Stationarity:')
get_stationarity(st_series, STATE), print()  # get data stationarity

get_ac_pac(st_series)  # pac & ac

print('## Autotests:')
autotests(st_series), print()  # adf, kpss, pp

##################################################################################################
# DATA SPLITTING
##################################################################################################

X = st_series
size = int(len(X) * 0.666)
train, test = X[0:size], X[size:len(X)]
print('TRAIN SIZE:')
print(len(train))
print('TEST SIZE:')
print(len(test)), print()

plt.plot(train)
plt.plot(test)
plt.title(STATE + ' train & test data division')  # train & test plot
plt.show()

get_ac_pac(st_series)


##################################################################################################
# FORECASTING
##################################################################################################

def forecasting(model, test_data, m_name):
    # Forecast
    n_periods = len(test_data)
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = test_data.index

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.plot(train, label='Training')
    plt.plot(test, label='Actual')
    plt.plot(fc_series.index, fc_series, color='darkgreen', label='Forecast')
    plt.legend(loc='best')
    plt.fill_between(lower_series.index,
                     lower_series,
                     upper_series,
                     color='k', alpha=.15)

    plt.title(m_name + ' - Final Forecast of ' + STATE + ' ILI RATIO')
    # plt.savefig('plots/' + STATE + '_' + m_name + '_forecast.png')
    plt.show()
    print('Forecast Accuracy: ')
    test_values = [x for x in test.ILI_RATIO]
    fc_values = [x for x in fc_series]
    test_values = np.asarray(test_values)
    fc_values = np.asarray(fc_values)

    rmsef, corrf = forecast_accuracy(fc_values, test_values)
    return rmsef, corrf


##################################################################################################
# ARIMA WITHOUT DATA SPLITTING
##################################################################################################

def normal_arima(series, p, d, q):
    print('################################################################################################## '
          'ARIMA WITHOUT DATA SPLITTING'
          ' ##################################################################################################')
    model = ARIMA(series, order=(p, d, q))
    results = model.fit()
    print(results.summary())

    residuals = pd.DataFrame(results.resid)
    residuals.plot(title="Residuals")
    plt.axhline(y=0, color='black', linestyle='--')
    plt.show()
    residuals.plot(kind='kde', title='Density')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.show()

    print()
    print('Residuals mean: ', residuals.mean())

    results.plot_diagnostics(figsize=(15, 12))  # 4 plot summary
    plt.savefig('plots/' + STATE + 'arimaDiag.png')
    plt.show()


# p_val, d_val, q_val = 1, 0, 0
# normal_arima(st_series, p_val, d_val, q_val)  # arima execution


##################################################################################################
# ARIMA (ROLLING FORECAST)
##################################################################################################

def arima_roll(train_series, test_series):
    print('################################################################################################## '
          'ARIMA (ROLLING FORECAST)'
          ' ##################################################################################################')
    history = [x for x in train_series.ILI_RATIO]
    predictions = list()

    for t in range(len(test_series)):
        model = ARIMA(history, order=(3, 0, 0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_series.ILI_RATIO[t]
        history.append(obs)
        if t % 10 == 0:
            print('predicted=%f, expected=%f' % (yhat, obs))

    # evaluate forecasts
    rmse = sqrt(mean_squared_error(test_series, predictions))
    print('Test RMSE: %.8f' % rmse)
    # plot forecasts against actual outcomes
    plt.plot(test_series, label='Acutal')
    plt.plot(test_series.index, predictions, color='red', label='Prediction')
    plt.legend(loc='best')
    plt.title('Rolling Forecast of ' + STATE + ' ILI RATIO')
    plt.savefig('plots/' + STATE + 'rollingFc.png')
    plt.show()
    print()

    test_values = [x for x in test.ILI_RATIO]
    fc_values = [x for x in predictions]
    actual = np.asarray(test_values)
    forecast = np.asarray(fc_values)
    rmse = np.mean((forecast - actual) ** 2) ** .5  # RMSE
    corr = np.corrcoef(forecast, actual)[0, 1]  # corr
    return rmse, corr


# rmseR, corrR = arima_roll(train, test)    # roll arima execution
# rmse_roll.append(rmseR)
# corr_roll.append(corrR)


##################################################################################################
# AUTO ARIMA
##################################################################################################

def autoarima(trains, tests):
    print('################################################################################################## '
          'AUTO ARIMA'
          ' ##################################################################################################')
    m_name = 'ARIMA'
    model = pm.auto_arima(trains, start_p=1, start_q=1,
                          test='adf',       # use adftest to find optimal 'd'
                          max_p=3, max_q=3, # maximum p and q
                          m=1,              # frequency of series
                          d=None,           # let model determine 'd'
                          seasonal=False,   # No Seasonality
                          start_P=0,
                          D=0,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)

    print(model.summary())

    model.plot_diagnostics(figsize=(15, 12))
    plt.savefig('plots/' + STATE + 'autoarimaDiag.png')
    plt.show()

    rmseAu, corrAu = forecasting(model, tests, m_name)
    return rmseAu, corrAu


rmseA, corrA = autoarima(train, test)    # auto arima execution

# rmse_auto.append(rmseA)
# corr_auto.append(corrA)


##################################################################################################
# SARIMA
##################################################################################################

def sarima(trains, tests):
    print('################################################################################################## '
          'SARIMA'
          ' ##################################################################################################')
    m_name = 'SARIMA'
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=100, sharex='all')

    # Usual Differencing
    axes[0].plot(trains[:], label='Original Series')
    axes[0].plot(trains[:].diff(1), label='Usual Differencing')
    axes[0].set_title('Usual Differencing')
    axes[0].legend(loc='upper left', fontsize=10)

    # Seasonal Differencing
    axes[1].plot(trains[:], label='Original Series')
    axes[1].plot(trains[:].diff(52), label='Seasonal Differencing', color='green')
    axes[1].set_title('Seasonal Differencing')
    plt.legend(loc='upper left', fontsize=10)
    plt.suptitle(STATE + ' ILI RATIO', fontsize=16)
    plt.savefig('plots/' + STATE + '_seasonal.png')
    plt.show()

    # Seasonal - fit stepwise auto-ARIMA
    smodel = pm.auto_arima(trains, start_p=1, start_q=1,
                           test='adf',
                           max_p=3, max_q=3, m=52,
                           d=None,
                           start_P=0, seasonal=True, D=1, trace=True,
                           error_action='ignore', suppress_warnings=True,
                           stepwise=True)

    print(smodel.summary())

    rmseSA, corrSA = forecasting(smodel, tests, m_name)
    return rmseSA, corrSA


rmseS, corrS = sarima(train, test)    # sarima execution

# rmse_seasonal.append(rmseS)
# corr_seasonal.append(corrS)


##################################################################################################
# RESULTS
##################################################################################################

rmse_roll = [0.007925465882889298, 0.00732408117436629, 0.004738959787475491, 0.00924158946395114, 0.0041241131662080755, 0.008118475845257376, 0.004571955303115961, 0.007200328876846231, 0.008975654450312254, 0.005107809419990095, 0.0024849384389337803, 0.005492774108943449, 0.003265746334566959, 0.0034558730325705777, 0.004626864318587985, 0.0069640100675736885, 0.0026421917560694, 0.00466401754931195, 0.006207374153295037, 0.005903309170032806, 0.0073158115039677575, 0.007020249906192543, 0.008382747339347175, 0.0032435275388968047, 0.005366962587464425, 0.0041440526344197195, 0.005153397215956519, 0.00861417595556335, 0.0037414449086804915, 0.007942342619493814, 0.00823851019046876, 0.004165940592732966, 0.004664204875147882, 0.005845557440696493, 0.004549852639115927, 0.005906957277634648, 0.004069291106763448]
corr_roll = [0.9520619809577452, 0.9086318844645738, 0.9346504203820604, 0.9293427488235094, 0.8584785057447273, 0.9494924076125327, 0.8275730963573467, 0.9659356299459729, 0.9500851207085084, 0.9683988179944887, 0.8485996659688281, 0.8993827810972018, 0.941215954261215, 0.9541439638065283, 0.9469527810305756, 0.9261179206696887, 0.9395942033687679, 0.8785708028487086, 0.9497908949595851, 0.923385333588008, 0.9528430862337048, 0.9246594686279919, 0.7261096669210775, 0.9306751862764313, 0.9044549756034722, 0.945711212497346, 0.9311089753062053, 0.9548299381246083, 0.9437401355148818, 0.9005986055840641, 0.9620496750117286, 0.888120756618014, 0.8647440876390998, 0.951471920350415, 0.8824041504159963, 0.9205115766954383, 0.9318781200032072]
rmse_auto = [0.026870164095099364, 0.020762660246547757, 0.015591261302201394, 0.02551489564340879, 0.008417602302660082, 0.028598825269665378, 0.011732284810022637, 0.02859209982478249, 0.0329088715791965, 0.020491230396083457, 0.004823456458786383, 0.01252266236904327, 0.009947030533786788, 0.012159824833643106, 0.014541341292213515, 0.017680369867757572, 0.008665372335121848, 0.010718132525188533, 0.02367352794668315, 0.01614514321157993, 0.025234081650075783, 0.018909427175707528, 0.013104031073617262, 0.009368261670042899, 0.012857157625986566, 0.013082519338334625, 0.014740564000463169, 0.03337662206754058, 0.011803120840561978, 0.01927986913948203, 0.030579487563521877, 0.011500720667114593, 0.010881969017246558, 0.01934658628143213, 0.009975438303287941, 0.015081274001388728, 0.011492486832393164]
corr_auto = [-0.007157154117782484, 0.023871240023521575, 0.10885040772293349, -0.05294888309536703, -0.010772310935628401, -0.16365377760728037, -0.018649884750199297, -0.029892926968783033, -0.08945809376735557, -0.1611367974811597, 0.25780173458836114, 0.09082350190725362, -0.10293486031016227, -0.18479158442975172, -0.09862760924648876, 0.22395062875288538, -0.20423985077197446, -0.00499387361321312, -1, -0.042513881529011116, -0.21442223421194756, 0.006145803233767491, 0.018438497584274148, -0.1029364020655269, -0.044016456622366, 0.03447080150741103, 0.0216660922217086, -0.15167769180990148, -0.09406511096312786, 0.05318271723259376, -0.115734736069683, 0.15029958443027083, -0.07077688682682268, -0.020776548982214232, -0.05989879123636427, 0.1313558581723483, 0.019512956459431196]
rmse_seasonal = [0.023224823502650573, 0.01865970907218524, 0.011153006465580217, 0.01757314661820364, 0.007274292584339075, 0.029696537006595868, 0.00575735065128208, 0.01986527415613189, 0.030222610508656822, 0.016667357699906262, 0.00772344427295225, 0.01128657683290528, 0.0073163357028527545, 0.008572691277226971, 0.010884604242161939, 0.026615042186059538, 0.00846359316956892, 0.009698044154156305, 0.016184104477399743, 0.01023433928761822, 0.024774926209885317, 0.014281325131300962, 0.0085379584744292, 0.008063253785262021, 0.01046359819351321, 0.009914068519153362, 0.011633757627475049, 0.029953772898746546, 0.008091629773198342, 0.014517227014043127, 0.021312325904098657, 0.011579625296913218, 0.008841521208507551, 0.016545206471850406, 0.006683363222685306, 0.01279337408007842, 0.008970879861990572]
corr_seasonal = [0.6926327872953565, 0.47626199858322843, 0.6340805684584675, 0.7624883853640919, 0.4416041351096932, 0.5257840792205642, 0.8225932872797221, 0.8251406385786832, 0.7038023216363346, 0.7004244820419847, 0.46454393124565485, 0.510528215070156, 0.728937522663975, 0.7051143910000967, 0.6480128824512359, 0.3154750601803115, 0.8224045849313872, 0.5466974262138683, 0.811158474336613, 0.768392024363527, 0.3325073589211812, 0.6744909186231729, 0.7055169877968114, 0.5793299396962734, 0.8781697595467561, 0.7275081129927685, 0.7727464506446176, 0.6834145168867994, 0.8359424197420035, 0.7549425876737982, 0.7385370135392915, 0.7054175595428132, 0.5678434283153028, 0.5959667563148386, 0.7735194496791515, 0.584391135503299, 0.6927493655348536]

print('states_abv len:', len(states_abv))
print()
print('ROLL RMSE len:', len(rmse_roll))
print('ROLL CORR len:', len(corr_roll))
print('AUTO RMSE len:', len(rmse_auto))
print('AUTO CORR len:', len(corr_auto))
print('SEASONAL RMSE len:', len(rmse_seasonal))
print('SEASONAL CORR len:', len(corr_seasonal))
print()
print("Best value ROLL RMSE: ", min(rmse_roll), 'state: ', rmse_roll.index(min(rmse_roll)), done_list[rmse_roll.index(min(rmse_roll))])
print("Best value ROLL CORR: ", max(corr_roll), 'state: ', corr_roll.index(max(corr_roll)), done_list[corr_roll.index(max(corr_roll))])
print("Best value AUTO RMSE: ", min(rmse_auto), 'state: ', rmse_auto.index(min(rmse_auto)), done_list[rmse_auto.index(min(rmse_auto))])
print("Best value AUTO CORR: ", max(corr_auto), 'state: ', corr_auto.index(max(corr_auto)), done_list[corr_auto.index(max(corr_auto))])
print("Best value SEASONAL RMSE: ", min(rmse_seasonal), 'state: ', rmse_seasonal.index(min(rmse_seasonal)), done_list[rmse_seasonal.index(min(rmse_seasonal))])
print("Best value SEASONAL CORR: ", max(corr_seasonal), 'state: ', corr_seasonal.index(max(corr_seasonal)), done_list[corr_seasonal.index(max(corr_seasonal))])
print()
print("Worst value ROLL RMSE: ", max(rmse_roll), 'state: ', rmse_roll.index(max(rmse_roll)), done_list[rmse_roll.index(max(rmse_roll))])
print("Worst value ROLL CORR: ", min(corr_roll), 'state: ', corr_roll.index(min(corr_roll)), done_list[corr_roll.index(min(corr_roll))])
print("Worst value AUTO RMSE: ", max(rmse_auto), 'state: ', rmse_auto.index(max(rmse_auto)), done_list[rmse_auto.index(max(rmse_auto))])
print("Worst value AUTO CORR: ", min(corr_auto), 'state: ', corr_auto.index(min(corr_auto)), done_list[corr_auto.index(min(corr_auto))])
print("Worst value SEASONAL RMSE: ", max(rmse_seasonal), 'state: ', rmse_seasonal.index(max(rmse_seasonal)), done_list[rmse_seasonal.index(max(rmse_seasonal))])
print("Worst value SEASONAL CORR: ", min(corr_seasonal), 'state: ', corr_seasonal.index(min(corr_seasonal)), done_list[corr_seasonal.index(min(corr_seasonal))])
print()
print('ROLL RMSE mean:', mean(rmse_roll))
print('ROLL CORR mean:', mean(corr_roll))
print('AUTO RMSE mean:', mean(rmse_auto))
print('AUTO CORR mean:', mean(corr_auto))
print('SEASONAL RMSE mean:', mean(rmse_seasonal))
print('SEASONAL CORR mean:', mean(corr_seasonal))

plt.rcParams.update({'figure.figsize': (12, 6), 'figure.dpi': 120})

plt.bar(states_abv, rmse_auto)
plt.title('ARIMA RMSE')
plt.xlabel('State')
plt.ylabel('RMSE')
plt.ylim(0, 0.04)
plt.show()
plt.bar(states_abv, corr_auto)
plt.title('ARIMA Correlation')
plt.xlabel('State')
plt.ylabel('Corr')
plt.ylim(0, 1)
plt.show()

plt.bar(states_abv, rmse_seasonal)
plt.title('SARIMA RMSE')
plt.xlabel('State')
plt.ylabel('RMSE')
plt.ylim(0, 0.04)
plt.show()
plt.bar(states_abv, corr_seasonal)
plt.title('SARIMA Correlation')
plt.xlabel('State')
plt.ylabel('Corr')
plt.ylim(0, 1)
plt.show()

'''
##################################################################################################
# COVID
##################################################################################################

def get_COVID_stationarity(data, country):
    # rolling statistics
    rolling_mean = data.rolling(window=10).mean()
    rolling_std = data.rolling(window=10).std()
    plt.plot(data, color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title(country + ' daily COVID-19 new cases rolling Mean & standard Deviation')
    # plt.savefig('covid/' + country + '_stationarity.png')
    plt.show(block=False)

    # Dickey–Fuller test:
    result = adfuller(data, autolag='AIC')
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
    if result[1] < 0.05:
        print('RESULT: This series is stationary.')
    else:
        print('RESULT: This series is non-stationary.')


df = pd.read_csv('WHO-COVID-19.csv', parse_dates=['Date_reported'], index_col=['Date_reported'])
df.info()

df = df[df.New_cases != 0]

df = df.drop(['Country_code', 'WHO_region', 'Cumulative_cases', 'New_deaths', 'Cumulative_deaths'], 1)
dfES = df[df.Country == 'Spain']
dfUS = df[df.Country == 'United States of America']
dfES.info()
dfUS.info()
dfES = dfES.drop(['Country'], 1)
dfUS = dfUS.drop(['Country'], 1)

print(dfES)
print(dfUS)

plt.xlabel('Date')
plt.ylabel('New cases')
plt.plot(dfES)
plt.title('Spain daily new cases from 2020 to 2022')
# plt.savefig('covid/Spain_newcases.png')
plt.show()

plt.xlabel('Date')
plt.ylabel('New cases')
plt.plot(dfUS)
plt.title('US daily new cases from 2020 to 2022')
# plt.savefig('covid/US_newcases.png')
plt.show()

print('## COVID Stationarity:')
get_COVID_stationarity(dfES, 'Spain'), print()  # get data stationarity

get_ac_pac(dfES)  # pac & ac

print('## Autotests:')
autotests(dfES), print()  # adf, kpss, pp

X = np.log(dfUS)
get_COVID_stationarity(X, 'Spainlog'), print()  # get data stationarity
autotests(X), print()  # adf, kpss, pp
get_ac_pac(X)

##################################################################################################
# DATA SPLITTING
##################################################################################################
COUNTRY = 'US'

X = np.log(dfUS)

plt.xlabel('Date')
plt.ylabel('New cases')
plt.plot(X)
plt.title('US daily new cases from 2020 to 2022 - log')
# plt.savefig('covid/US_newcases_log.png')
plt.show()

size = int(len(X) * 0.666)
train, test = X[0:size], X[size:len(X)]
print('TRAIN SIZE:')
print(len(train))
print('TEST SIZE:')
print(len(test)), print()

plt.plot(train)
plt.plot(test)
plt.title(COUNTRY + ' train & test data division')  # train & test plot
plt.show()


def COVIDforecasting(model, test_data, m_name):
    # Forecast
    n_periods = len(test_data)
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = test_data.index

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.plot(train, label='Training')
    plt.plot(test, label='Actual')
    plt.plot(fc_series.index, fc_series, color='darkgreen', label='Forecast')
    plt.legend(loc='best')
    plt.fill_between(lower_series.index,
                     lower_series,
                     upper_series,
                     color='k', alpha=.15)

    plt.title(m_name + ' - Final Forecast of ' + COUNTRY + ' new COVID-19 cases')
    # plt.savefig('covid/' + COUNTRY + '_' + m_name + '_forecast.png')
    plt.show()
    print('Forecast Accuracy: ')
    test_values = [x for x in test.New_cases]
    fc_values = [x for x in fc_series]
    test_values = np.asarray(test_values)
    fc_values = np.asarray(fc_values)

    rmsef, corrf = forecast_accuracy(fc_values, test_values)
    return rmsef, corrf


model = pm.auto_arima(train, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

print(model.summary())

model.plot_diagnostics(figsize=(15, 12))
# plt.savefig('covid/' + COUNTRY + 'autoarimaDiag.png')
plt.show()

rmsecov, corrcov = COVIDforecasting(model, test, 'ARIMA')
'''