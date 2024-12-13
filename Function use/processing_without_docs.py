# -*- coding: utf-8 -*-


# Function to read sheets from an Excel file
def read_excel_sheets(file_path: str, sheets: list) -> Dict[str, pd.DataFrame]:
    """
    Read data from specified sheets in an Excel file into a dictionary of dataframes.

    Parameters:
    file_path (str): Path to the Excel file.
    sheets (list): A list of sheet names to be read from the Excel file.

    Returns:
    Dict[str, pd.DataFrame]: A dictionary where keys are sheet names and values are the corresponding dataframes.
    """

    # Read the Excel file and store each sheet in a dictionary
    dataframes = {}
    for sheet in sheets:
        dataframes[sheet] = pd.read_excel(file_path, sheet_name=sheet)

    return dataframes

# Function to calculate descriptive statistics and normality test
def calculate_statistics(dataframes: dict) -> pd.DataFrame:
    """
    Calculate descriptive statistics and normality test for each sheet.

    Parameters:
    - dataframes (dict): Dictionary with sheet names as keys and dataframes as values.

    Returns:
    - pd.DataFrame: DataFrame with descriptive statistics and normality test results.
    """
    stats_dict = {
        'Variable': [],
        'Mean': [],
        'Std': [],
        'Min': [],
        'Max': [],
        'Nb Observations': [],
        'Normality (p-value)': []
    }

    for sheet, df in dataframes.items():
        usd_values = df['USD'].dropna()  # Drop NA values for accurate statistics

        # Calculate statistics
        mean_val = usd_values.mean()
        std_val = usd_values.std()
        min_val = usd_values.min()
        max_val = usd_values.max()
        n_obs = len(usd_values)

        # Normality test (Shapiro-Wilk test)
        shapiro_test = stats.shapiro(usd_values)
        normality_p = shapiro_test.pvalue

        # Append results
        stats_dict['Variable'].append(sheet)
        stats_dict['Mean'].append(mean_val)
        stats_dict['Std'].append(std_val)
        stats_dict['Min'].append(min_val)
        stats_dict['Max'].append(max_val)
        stats_dict['Nb Observations'].append(n_obs)
        stats_dict['Normality (p-value)'].append(normality_p)

    # Create a DataFrame from the dictionary
    stats_df = pd.DataFrame(stats_dict)
    return stats_df

def wavelet_decomposition_and_plot(sheet_name, dataframes, level=6, wavelet_name='db6', mode='sym', save_fig=False):
    # Extract the data from the specified sheet
    data = dataframes[sheet_name]

    # Ensure the time index is set
    data.set_index('temps', inplace=True)

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(data, wavelet_name, mode, level=level)  # Ensure data is a series .squeeze()
    (cA, *cD) = coeffs


    # Plotting
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1], width_ratios=[2, 1])
    axs = [fig.add_subplot(gs[0, :]),  # Original series spanning both columns
           fig.add_subplot(gs[1, 0]),  # Approximation coefficients
           fig.add_subplot(gs[1, 1]),  # Empty placeholder
           fig.add_subplot(gs[2, 0]),  # Detail coefficients
           fig.add_subplot(gs[2, 1]),  # Detail coefficients
           fig.add_subplot(gs[3, 0]),  # Detail coefficients
           fig.add_subplot(gs[3, 1])]  # Detail coefficients

    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    # Plot the original data spanning across both columns at the top
    axs[0].plot(data)
    #axs[0].set_ylabel(f's ({sheet_name})', fontsize=13)
    axs[0].set_title(f'{sheet_name}', fontsize=14)
    axs[0].grid(True)

    # Plot the approximation coefficients
    axs[1].plot(cA)
    axs[1].set_ylabel(f'cA{level}', fontsize=13)
    axs[1].set_title(f'cA{level}', fontsize=14)
    axs[1].grid(True)

    # Empty subplot in the second row
    #axs[2].axis('off')

    # Plot the detail coefficients cD1 to cD6
    for i in range(level):
        axs[6 - i].plot(cD[level - 1 - i])
        axs[6 - i].set_ylabel(f'cD{level - i}', fontsize=13)
        axs[6 - i].set_title(f'cD{level - i}', fontsize=14)
        axs[6 - i].grid(True)

    # Set common labels for the x-axis
    axs[5].set_xlabel('Index', fontsize=13)
    axs[6].set_xlabel('Index', fontsize=13)

    plt.tight_layout()

    # Save the figure if needed
    if save_fig:
        plt.savefig(f'{sheet_name}_coeffs.png', dpi=300)

    plt.show()

# Function to select scales based on the time level
def select_scales(time_level: str) -> np.ndarray:
    if time_level == 'yearly':
        return np.arange(1, 32)  # Smaller scales for yearly data 32
    elif time_level == 'quarterly':
        return np.arange(1, 64)  # Medium scales for quarterly data 64
    elif time_level == 'monthly':
        return np.arange(1, 128)  # Larger scales for monthly data 128
    elif time_level == 'weekly':
        return np.arange(1, 256)  # Even larger scales for weekly data 256
    elif time_level == 'daily':
        return np.arange(1, 2028)  # Largest scales for daily data 512  1024

# Function to format date labels based on time frequency
def format_date_labels(ax, time_level: str):
    if time_level == 'yearly':
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    elif time_level == 'quarterly':
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-Q%q'))
    elif time_level == 'monthly':
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    elif time_level == 'weekly':
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%W'))
    elif time_level == 'daily':
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

# Function to perform wavelet analysis and plot results using contourf for multiple wavelets
def wavelet_analysis_and_plot(file_path: str, sheets: list, time_levels: list, wavelet_types: list):
    dataframes = read_excel_sheets(file_path, sheets)

    # Number of rows and columns for subplots
    nrows = len(sheets)
    ncols = len(wavelet_types) + 1  # +1 for the time series plot

    # Create a figure for subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))

    for i, sheet in enumerate(sheets):
        df = dataframes[sheet]
        time_level = time_levels[i]

        # Convert the 'temps' column to datetime
        if time_level == 'yearly':
            df['temps'] = pd.to_datetime(df['temps'], format='%Y')
        elif time_level == 'quarterly':
            df['temps'] = pd.to_datetime(df['temps'], format='%Y-Q%q')
        elif time_level == 'monthly':
            df['temps'] = pd.to_datetime(df['temps'], format='%Y-%m')
        elif time_level == 'weekly':
            df['temps'] = pd.to_datetime(df['temps'], format='%Y-%W')
        elif time_level == 'daily':
            df['temps'] = pd.to_datetime(df['temps'], format='%Y-%m-%d')

        # Extract the time series (USD column)
        time_series = df['USD'].values

        # Plot the time series in the first column
        axs[i, 0].plot(df['temps'], time_series, label='USD', color='blue')
        axs[i, 0].set_title(f'Time Series - {sheet}')
        axs[i, 0].set_xlabel('Time')
        axs[i, 0].set_ylabel('USD')
        axs[i, 0].legend()
        axs[i, 0].grid(True)

        # Format the date labels based on the time level
        format_date_labels(axs[i, 0], time_level)

        # Perform wavelet analysis for each wavelet type
        for j, wavelet in enumerate(wavelet_types):
            # Select appropriate scales based on the time level
            scales = select_scales(time_level)

            # Perform the continuous wavelet transform using the specified wavelet
            coefficients, frequencies = pywt.cwt(time_series, scales, wavelet)

            # Time axis for contour plot
            time_axis = np.arange(len(time_series))

            # Plot the wavelet power spectrum using contourf
            contour = axs[i, j + 1].contourf(time_axis, np.log2(frequencies), np.abs(coefficients) ** 2, levels=100, cmap='jet')
            fig.colorbar(contour, ax=axs[i, j + 1], label='Power')

            axs[i, j + 1].set_title(f'Wavelet: {wavelet} - {sheet}')
            axs[i, j + 1].set_xlabel('Time')
            axs[i, j + 1].set_ylabel('Scale (Log Frequency)')

            # Format the date labels on the wavelet plot as well
            format_date_labels(axs[i, j + 1], time_level)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def detect_regimes(coefficients, scales):
    power = np.abs(coefficients) ** 2
    avg_power = np.mean(power, axis=0)
    threshold = avg_power.mean() + 2 * avg_power.std()
    regimes = []
    in_regime = False
    start = 0

    for t in range(len(avg_power)):
        if avg_power[t] > threshold and not in_regime:
            start = t
            in_regime = True
        elif avg_power[t] <= threshold and in_regime:
            regimes.append((start, t))
            in_regime = False

    return regimes

def simple_regime_detection(time_series, window_size=10):
    # Calculate rolling mean and variance
    rolling_mean = pd.Series(time_series).rolling(window=window_size).mean().dropna()
    rolling_var = pd.Series(time_series).rolling(window=window_size).var().dropna()

    # Simple segmentation based on rolling mean and variance
    mean_threshold = rolling_mean.mean()
    var_threshold = rolling_var.mean()

    regimes = (rolling_mean > mean_threshold) & (rolling_var > var_threshold)
    regimes = regimes.astype(int)

    return regimes

# Function to perform wavelet analysis and plot results using contourf for multiple wavelets
def wavelet_regime_and_plot(file_path: str, sheets: list, time_levels: list, wavelet_types: list):
    dataframes = read_excel_sheets(file_path, sheets)

    # Number of rows and columns for subplots
    nrows = len(sheets)
    ncols = len(wavelet_types) + 1  # +1 for the time series plot

    # Create a figure for subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))

    for i, sheet in enumerate(sheets):
        df = dataframes[sheet]
        time_level = time_levels[i]

        # Convert the 'temps' column to datetime
        if time_level == 'yearly':
            df['temps'] = pd.to_datetime(df['temps'], format='%Y')
        elif time_level == 'quarterly':
            df['temps'] = pd.to_datetime(df['temps'], format='%Y-Q%q')
        elif time_level == 'monthly':
            df['temps'] = pd.to_datetime(df['temps'], format='%Y-%m')
        elif time_level == 'weekly':
            df['temps'] = pd.to_datetime(df['temps'], format='%Y-%W')
        elif time_level == 'daily':
            df['temps'] = pd.to_datetime(df['temps'], format='%Y-%m-%d')

        # Extract the time series (USD column)
        time_series = df['USD'].values

        # Plot the time series in the first column
        axs[i, 0].plot(df['temps'], time_series, label='USD', color='blue')
        axs[i, 0].set_title(f'Time Series - {sheet}')
        axs[i, 0].set_xlabel('Time')
        axs[i, 0].set_ylabel('USD')
        axs[i, 0].legend()
        axs[i, 0].grid(True)

        # Format the date labels based on the time level
        format_date_labels(axs[i, 0], time_level)

        # Perform wavelet analysis for each wavelet type
        for j, wavelet in enumerate(wavelet_types):
            # Select appropriate scales based on the time level
            scales = select_scales(time_level)

            # Perform the continuous wavelet transform using the specified wavelet
            coefficients, frequencies = pywt.cwt(time_series, scales, wavelet)

            # Time axis for contour plot
            time_axis = np.arange(len(time_series))

            # Plot the wavelet power spectrum using contourf
            contour = axs[i, j + 1].contourf(time_axis, np.log2(frequencies), np.abs(coefficients) ** 2, levels=100, cmap='jet')
            fig.colorbar(contour, ax=axs[i, j + 1], label='Power')

            axs[i, j + 1].set_title(f'Wavelet: {wavelet} - {sheet}')
            axs[i, j + 1].set_xlabel('Time')
            axs[i, j + 1].set_ylabel('Scale (Log Frequency)')

            # Format the date labels on the wavelet plot as well
            format_date_labels(axs[i, j + 1], time_level)

            # Detect regimes based on wavelet power spectrum
            regimes = detect_regimes(coefficients, scales)

            # Highlight detected regimes in the time series plot
            for start, end in regimes:
                axs[i, 0].axvspan(df['temps'].iloc[start], df['temps'].iloc[end], color='red', alpha=0.3)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def clustering_analysis(file_path: str, sheets: list, time_levels: list, n_clusters: int = 2):
    dataframes = read_excel_sheets(file_path, sheets)

    # Create a figure for subplots
    fig, axs = plt.subplots(nrows=len(sheets), figsize=(10, 6 * len(sheets)))

    for i, sheet in enumerate(sheets):
        df = dataframes[sheet]
        time_level = time_levels[i]

        # Convert the 'temps' column to datetime
        df['temps'] = pd.to_datetime(df['temps'])

        # Extract the time series (USD column)
        time_series = df[['USD']].values

        # Standardize the time series
        scaler = StandardScaler()
        time_series_scaled = scaler.fit_transform(time_series)

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=n_clusters)
        df['Cluster'] = kmeans.fit_predict(time_series_scaled)

        # Plot the time series and clusters
        axs[i].plot(df['temps'], time_series, label='USD', color='blue')
        for cluster in range(n_clusters):
            axs[i].fill_between(df['temps'], min(time_series_scaled), max(time_series_scaled),
                                where=(df['Cluster'] == cluster),
                                color=f'C{cluster+1}', alpha=0.3, label=f'Cluster {cluster+1}')

        axs[i].set_title(f'Clustering Analysis - {sheet}')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('USD')
        axs[i].legend()
        axs[i].grid(True)

        # Format the date labels based on the time level
        format_date_labels(axs[i], time_level)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

#
def unified_wavelet_clustering_analysis(file_path: str, sheets: list, time_levels: list, wavelet_type: str, n_clusters: int = 3):
    dataframes = read_excel_sheets(file_path, sheets)

    # Create a figure for subplots
    fig, axs = plt.subplots(nrows=len(sheets), ncols=3, figsize=(15, 6 * len(sheets)))

    for i, sheet in enumerate(sheets):
        df = dataframes[sheet]
        time_level = time_levels[i]

        # Convert the 'temps' column to datetime
        df['temps'] = pd.to_datetime(df['temps'])

        # Extract the time series (USD column)
        time_series = df['USD'].values

        # Perform wavelet analysis
        scales = select_scales(time_level)
        coefficients, frequencies = pywt.cwt(time_series, scales, wavelet_type)
        wavelet_power = np.abs(coefficients) ** 2

        # Average power spectrum across scales for simple regime detection
        avg_power = np.mean(wavelet_power, axis=0)

        # Simple regime detection based on rolling statistics
        regimes = simple_regime_detection(time_series, window_size=10)

        # Apply clustering on the wavelet coefficients
        scaler = StandardScaler()
        coefficients_scaled = scaler.fit_transform(np.transpose(coefficients))
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(coefficients_scaled)

        # Plot the wavelet power spectrum
        contour = axs[i, 0].contourf(np.arange(len(avg_power)), np.log2(frequencies), wavelet_power, levels=100, cmap='jet')
        fig.colorbar(contour, ax=axs[i, 0], label='Power')
        axs[i, 0].set_title(f'Wavelet Spectrum - {wavelet_type} - {sheet}')
        axs[i, 0].set_xlabel('Time')
        axs[i, 0].set_ylabel('Scale (Log Frequency)')
        format_date_labels(axs[i, 0], time_level)

        # Plot the original time series with detected regimes
        axs[i, 1].plot(df['temps'], time_series, label='USD', color='blue')
        axs[i, 1].fill_between(df['temps'][len(df['temps']) - len(regimes):], min(time_series), max(time_series),
                               where=regimes, color='red', alpha=0.3, label='Detected Regime')
        axs[i, 1].set_title(f'Regime Detection - {sheet}')
        axs[i, 1].set_xlabel('Time')
        axs[i, 1].set_ylabel('USD')
        axs[i, 1].legend()
        axs[i, 1].grid(True)
        format_date_labels(axs[i, 1], time_level)

        # Plot the time series with cluster labels
        axs[i, 2].plot(df['temps'], time_series, label='USD', color='blue')
        for cluster in range(n_clusters):
            where = (cluster_labels == cluster)
            axs[i, 2].scatter(df['temps'][where], time_series[where], label=f'Cluster {cluster+1}', s=10)
        axs[i, 2].set_title(f'Clustering - {sheet}')
        axs[i, 2].set_xlabel('Time')
        axs[i, 2].set_ylabel('USD')
        axs[i, 2].legend()
        axs[i, 2].grid(True)
        format_date_labels(axs[i, 2], time_level)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

#
def unified_wavelet_clustering_analysis_a(file_path: str, sheets: list, time_levels: list, wavelet_type: str, n_clusters: int = 2):
    dataframes = read_excel_sheets(file_path, sheets)

    # Create a figure for subplots
    fig, axs = plt.subplots(nrows=len(sheets), ncols=3, figsize=(15, 6 * len(sheets)))

    for i, sheet in enumerate(sheets):
        df = dataframes[sheet]
        time_level = time_levels[i]

        # Convert the 'temps' column to datetime
        df['temps'] = pd.to_datetime(df['temps'])

        # Extract the time series (USD column)
        time_series = df['USD'].values

        # Perform wavelet analysis
        scales = select_scales(time_level)
        coefficients, frequencies = pywt.cwt(time_series, scales, wavelet_type)
        wavelet_power = np.abs(coefficients) ** 2

        # Average power spectrum across scales for simple regime detection
        avg_power = np.mean(wavelet_power, axis=0)

        # Simple regime detection based on rolling statistics
        regimes = simple_regime_detection(time_series, window_size=10)

        # Apply clustering on the wavelet coefficients
        scaler = StandardScaler()
        coefficients_scaled = scaler.fit_transform(np.transpose(coefficients))
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(coefficients_scaled)

        # Plot the wavelet power spectrum
        contour = axs[i, 0].contourf(np.arange(len(avg_power)), np.log2(frequencies), wavelet_power, levels=100, cmap='jet')
        fig.colorbar(contour, ax=axs[i, 0], label='Power')
        axs[i, 0].set_title(f'Wavelet Spectrum - {wavelet_type} - {sheet}')
        axs[i, 0].set_xlabel('Time')
        axs[i, 0].set_ylabel('Scale (Log Frequency)')
        format_date_labels(axs[i, 0], time_level)

        # Plot the original time series with detected regimes
        axs[i, 1].plot(df['temps'], time_series, label='USD', color='blue')
        axs[i, 1].fill_between(df['temps'][len(df['temps']) - len(regimes):], min(time_series), max(time_series),
                               where=regimes, color='red', alpha=0.3, label='Detected Regime')
        axs[i, 1].set_title(f'Regime Detection - {sheet}')
        axs[i, 1].set_xlabel('Time')
        axs[i, 1].set_ylabel('USD')
        axs[i, 1].legend()
        axs[i, 1].grid(True)
        format_date_labels(axs[i, 1], time_level)

        # Plot the time series with cluster labels
        axs[i, 2].plot(df['temps'], time_series, label='USD', color='blue')
        for cluster in range(n_clusters):
            where = (cluster_labels == cluster)
            axs[i, 2].scatter(df['temps'][where], time_series[where], label=f'Cluster {cluster+1}', s=10)
        axs[i, 2].set_title(f'Clustering - {sheet}')
        axs[i, 2].set_xlabel('Time')
        axs[i, 2].set_ylabel('USD')
        axs[i, 2].legend()
        axs[i, 2].grid(True)
        format_date_labels(axs[i, 2], time_level)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def extract_breakpoints(dates, labels):
    """
    Extract dates corresponding to changes in regime or cluster labels.

    Parameters:
    dates (pd.Series): A pandas Series containing the dates.
    labels (np.ndarray): An array of regime or cluster labels.

    Returns:
    pd.Series: Dates where regime or cluster labels change.
    """
    breakpoints_indices = np.where(labels[1:] != labels[:-1])[0] + 1
    breakpoints_dates = dates.iloc[breakpoints_indices]
    return breakpoints_dates

def unified_wavelet_clustering_analysis(file_path: str, sheets: list, time_levels: list, wavelet_types: list, n_clusters: int):
    dataframes = read_excel_sheets(file_path, sheets)

    nrows = len(sheets)
    ncols = len(wavelet_types) + 2  # +2 for the time series plot and clustering plot

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))

    for i, sheet in enumerate(sheets):
        df = dataframes[sheet]
        time_level = time_levels[i]

        if time_level == 'yearly':
            df['temps'] = pd.to_datetime(df['temps'], format='%Y')
        elif time_level == 'quarterly':
            df['temps'] = pd.to_datetime(df['temps'], format='%Y-Q%q')
        elif time_level == 'monthly':
            df['temps'] = pd.to_datetime(df['temps'], format='%Y-%m')
        elif time_level == 'weekly':
            df['temps'] = pd.to_datetime(df['temps'], format='%Y-%W')
        elif time_level == 'daily':
            df['temps'] = pd.to_datetime(df['temps'], format='%Y-%m-%d')

        time_series = df['USD'].values

        axs[i, 0].plot(df['temps'], time_series, label='USD', color='blue')
        axs[i, 0].set_title(f'Time Series - {sheet}')
        axs[i, 0].set_xlabel('Time')
        axs[i, 0].set_ylabel('USD')
        axs[i, 0].legend()
        axs[i, 0].grid(True)
        format_date_labels(axs[i, 0], time_level)

        # Perform wavelet analysis for each wavelet type
        for j, wavelet in enumerate(wavelet_types):
            scales = select_scales(time_level)
            coefficients, frequencies = pywt.cwt(time_series, scales, wavelet)
            time_axis = np.arange(len(time_series))
            contour = axs[i, j + 1].contourf(time_axis, np.log2(frequencies), np.abs(coefficients) ** 2, levels=100, cmap='jet')
            fig.colorbar(contour, ax=axs[i, j + 1], label='Power')
            axs[i, j + 1].set_title(f'Wavelet: {wavelet} - {sheet}')
            axs[i, j + 1].set_xlabel('Time')
            axs[i, j + 1].set_ylabel('Scale (Log Frequency)')
            format_date_labels(axs[i, j + 1], time_level)

        # Clustering on the original time series
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(time_series.reshape(-1, 1))
        axs[i, -1].plot(df['temps'], labels, color='green')
        axs[i, -1].set_title(f'Clustering - {sheet}')
        axs[i, -1].set_xlabel('Time')
        axs[i, -1].set_ylabel('Cluster')
        axs[i, -1].grid(True)

        # Extract and print breakpoints
        breakpoints_dates = extract_breakpoints(df['temps'], labels)
        print(f"Breakpoints for {sheet} occur on the following dates:")
        print(breakpoints_dates)

    plt.tight_layout()
    plt.show()

# Descriptive statistics
def calculate_descriptive_statistics(data_dicts, sheets_to_read):
    # Initialize a list to hold the results
    results = []

    # Iterate over each dataframe type
    for df_name, dataframes in data_dicts.items():
        # Iterate over each sheet name
        for sheet in sheets_to_read:
            if sheet not in dataframes:
                continue

            df = dataframes[sheet]

            if 'USD' not in df.columns:
                print(f"'USD' column not found in {sheet} of {df_name}. Skipping.")
                continue

            time_series = df['USD'].dropna()  # Drop missing values if any

            # Calculate descriptive statistics
            mean_val = time_series.mean()
            min_val = time_series.min()
            max_val = time_series.max()
            std_val = time_series.std()
            nb_obs = time_series.count()

            # Perform Shapiro-Wilk test for normality
            stat, p_value = shapiro(time_series)
            normality_p_val = f"{p_value:.3f}"

            if p_value <= 0.01:
                normality_p_val += ' ***'
            elif p_value <= 0.05:
                normality_p_val += ' **'
            elif p_value <= 0.10:
                normality_p_val += ' *'

            # Compile results
            results.append({
                'DataFrame': df_name,
                'Sheet': sheet,
                'Mean': round(mean_val, 3),
                'Min': round(min_val, 3),
                'Max': round(max_val, 3),
                'Std': round(std_val, 3),
                'Observations': nb_obs,
                'Normality (p-value)': normality_p_val
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Pivot the table to get 'Sheet' as columns and statistics as rows
    results_df_pivot = results_df.pivot(index='Sheet', columns='DataFrame').T

    return results_df_pivot

# Stationarity (URT)
def adf_test(timeseries):
    result = adfuller(timeseries)
    return result[0], result[1]

def kpss_test(timeseries):
    result = kpss(timeseries, regression='c')
    return result[0], result[1]

def plot_rolling_statistics(ax, timeseries, title, window=12):
    rolmean = timeseries.rolling(window=window).mean()
    rolstd = timeseries.rolling(window=window).std()

    ax.plot(timeseries, color='blue', label='Original Series')
    ax.plot(rolmean, color='red', label='Rolling Mean')
    ax.plot(rolstd, color='black', label='Rolling Std')
    ax.legend(loc='best')
    ax.set_title(title)
    ax.grid(True)  # Add grid lines

def analyze_stationarity(dataframes1, dataframes2, dataframes_10):
    # Initialize an empty list to store test results
    results = []

    # Prepare data structure to iterate through the three sets of dataframes
    datasets = {'dataframes': dataframes1, 'dataframes_79': dataframes2, 'dataframes_10': dataframes_10}

    # Determine subplot grid dimensions
    nrows = len(dataframes1)
    ncols = 3  # Three columns for the three different datasets

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

    # If there's only one row, axs will not be a list of lists; make it so for consistency
    if nrows == 1:
        axs = [axs]

    # Iterate over each sheet and each dataset
    for i, sheet in enumerate(dataframes1.keys()):
        row_results = {'Sheet': sheet}

        for j, (dataset_name, dataset) in enumerate(datasets.items()):
            if sheet not in dataset:
                continue
            df = dataset[sheet]

            if 'USD' not in df.columns:
                print(f"'USD' column not found in {sheet} of {dataset_name}. Skipping.")
                continue

            time_series = df['USD'].dropna()  # Drop missing values if any

            # Perform ADF and KPSS tests
            adf_stat, adf_p = adf_test(time_series)
            kpss_stat, kpss_p = kpss_test(time_series)

            # Format p-values with three decimal places and add significance markers
            adf_p_str = f"{adf_p:.3f}"
            kpss_p_str = f"{kpss_p:.3f}"

            if adf_p <= 0.01:
                adf_p_str += ' ***'
            elif adf_p <= 0.05:
                adf_p_str += ' **'
            elif adf_p <= 0.10:
                adf_p_str += ' *'

            if kpss_p <= 0.01:
                kpss_p_str += ' ***'
            elif kpss_p <= 0.05:
                kpss_p_str += ' **'
            elif kpss_p <= 0.10:
                kpss_p_str += ' *'

            # Append results to the row result
            row_results[f'{dataset_name} ADF Statistic'] = f"{adf_stat:.3f}"
            row_results[f'{dataset_name} ADF p-value'] = adf_p_str
            row_results[f'{dataset_name} KPSS Statistic'] = f"{kpss_stat:.3f}"
            row_results[f'{dataset_name} KPSS p-value'] = kpss_p_str

            # Plot Rolling Mean, Rolling Std, and Original Series in subplots
            plot_rolling_statistics(axs[i][j], time_series, title=f'{sheet} - {dataset_name}')

        # Append the row result to results
        results.append(row_results)

    plt.tight_layout()

    # Save the figure as a JPEG image
    plt.savefig('URT.jpeg', format='jpeg')

    plt.show()

    # Convert results list to a DataFrame and transpose
    results_df = pd.DataFrame(results)
    results_df.set_index('Sheet', inplace=True)  # Set 'Sheet' as index

    # Transpose the DataFrame to have sheets as columns
    results_df = results_df.T

    return results_df



