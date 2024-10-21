import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

# DataLoader Class - Handles data loading and file input
class DataLoader:
    def __init__(self, file_paths):
        """
        Initialize with either a single file path or a list of file paths.
        """
        if isinstance(file_paths, str):
            self.file_paths = [os.path.join('stock_data', file_paths)]  # Corrected single file handling
        else:
            self.file_paths = [os.path.join('stock_data', file) for file in file_paths]  # Handle list of files
        self.data = None

    def load_data(self):
        """
        Load financial data from one or more CSV files into a pandas DataFrame.
        If multiple files are provided, merge them on the 'Date' column.
        """
        if not all(os.path.exists(file) for file in self.file_paths):
            raise FileNotFoundError("One or more files not found.")

        if len(self.file_paths) == 1:
            self.data = pd.read_csv(self.file_paths[0])
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            print("Stock data loaded successfully.")
        else:
            file_names = [os.path.splitext(f)[0] for f in self.file_paths]
            combined_data = pd.read_csv(self.file_paths[0])
            combined_data = combined_data[['Date', 'Adj Close']]
            combined_data.rename(columns={'Adj Close': f'Adj Close_{file_names[0]}'}, inplace=True)

            for i in range(1, len(self.file_paths)):
                stock_data = pd.read_csv(self.file_paths[i])
                stock_data = stock_data[['Date', 'Adj Close']]
                stock_data.rename(columns={'Adj Close': f'Adj Close_{file_names[i]}'}, inplace=True)
                combined_data = pd.merge(combined_data, stock_data, on='Date', how='inner')

            self.data = combined_data
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            print("Stock data loaded and merged successfully.")
        
        return self.data

# MergeSort Class - Responsible for sorting the data using Merge Sort
class MergeSort:
    def __init__(self, key_column):
        """
        Initialize the MergeSort class with a specific key column.
        """
        self.key_column = key_column

    def merge_sort(self, arr):
        """
        Perform merge sort on a list of dictionaries (rows of a DataFrame).
        """
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        # Recursively sort both halves
        left_sorted = self.merge_sort(left_half)
        right_sorted = self.merge_sort(right_half)

        # Merge the sorted halves
        return self.merge(left_sorted, right_sorted)

    def merge(self, left, right):
        """
        Merge two sorted lists into one sorted list based on the key_column.
        """
        sorted_list = []
        i = j = 0

        while i < len(left) and j < len(right):
            if left[i][self.key_column] <= right[j][self.key_column]:
                sorted_list.append(left[i])
                i += 1
            else:
                sorted_list.append(right[j])
                j += 1

        while i < len(left):
            sorted_list.append(left[i])
            i += 1

        while j < len(right):
            sorted_list.append(right[j])
            j += 1

        return sorted_list

    def sort(self, data):
        """
        Main function to initiate sorting using merge sort.
        """
        data_list = data.to_dict('records')
        sorted_data = self.merge_sort(data_list)
        return pd.DataFrame(sorted_data)

class MaxSubarray:
    @staticmethod
    def max_crossing_sum(arr, low, mid, high):
        """
        Find the maximum sum of the subarray crossing the middle point.
        """
        left_sum = float('-inf')
        total = 0

        for i in range(mid, low - 1, -1):
            total += arr[i]
            left_sum = max(left_sum, total)

        right_sum = float('-inf')
        total = 0
        for i in range(mid + 1, high + 1):
            total += arr[i]
            right_sum = max(right_sum, total)

        return left_sum + right_sum

    @staticmethod
    def max_subarray_sum(arr, low, high):
        """
        Find the maximum subarray sum within a specified range using the divide-and-conquer approach.
        """
        if low == high:
            return arr[low], low, high

        mid = (low + high) // 2

        left_sum, left_start, left_end = MaxSubarray.max_subarray_sum(arr, low, mid)
        right_sum, right_start, right_end = MaxSubarray.max_subarray_sum(arr, mid + 1, high)
        cross_sum = MaxSubarray.max_crossing_sum(arr, low, mid, high)

        if left_sum >= right_sum and left_sum >= cross_sum:
            return left_sum, left_start, left_end
        elif right_sum >= left_sum and right_sum >= cross_sum:
            return right_sum, right_start, right_end
        else:
            return cross_sum, left_start, right_end

    @staticmethod
    def find_max_gain(prices):
        """
        Calculate the maximum gain from a list of price changes.
        """
        price_changes = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
        max_gain, start_idx, end_idx = MaxSubarray.max_subarray_sum(price_changes, 0, len(price_changes) - 1)
        
        return max_gain, start_idx, end_idx + 1

    @staticmethod
    def max_submatrix(matrix):
        """
        Find the maximum subarray sum in a 2D array (matrix) using divide and conquer.
        This will return the maximum sum and the coordinates of the submatrix.
        """
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return 0, 0, 0, 0, 0
        
        rows, cols = len(matrix), len(matrix[0])
        max_sum = float('-inf')
        top, bottom, left, right = 0, 0, 0, 0

        for left_col in range(cols):
            temp = [0] * rows
            for right_col in range(left_col, cols):
                for row in range(rows):
                    temp[row] += matrix[row][right_col]

                current_sum, current_start, current_end = MaxSubarray.max_subarray_sum(temp, 0, len(temp) - 1)

                if current_sum > max_sum:
                    max_sum = current_sum
                    top, bottom = current_start, current_end
                    left, right = left_col, right_col

        return max_sum, top, bottom, left, right

class SpikeDipClosestPair:
    @staticmethod
    def find_all_significant_spikes_dips(prices, threshold):
        """
        Main function to find all significant spikes or dips in the shortest time using divide-and-conquer.
        """
        prices.sort(key=lambda point: point[0])  
        return SpikeDipClosestPair._find_all_changes_rec(prices, threshold)

    @staticmethod
    def _find_all_changes_rec(prices, threshold):
        """
        Recursive function to find all significant spikes/dips per day between points based on prices.
        """
        n = len(prices)
        if n <= 3:
            return SpikeDipClosestPair.brute_force(prices, threshold)

        mid = n // 2
        mid_point = prices[mid]

        left_half = prices[:mid]
        right_half = prices[mid:]

        left_changes = SpikeDipClosestPair._find_all_changes_rec(left_half, threshold)
        right_changes = SpikeDipClosestPair._find_all_changes_rec(right_half, threshold)

        strip = []
        for price in prices:
            if abs((price[0]) - mid_point[0]).days < 10:
                strip.append(price)

        strip_changes = SpikeDipClosestPair.strip_collect_significant_changes(strip, threshold)
        
        return left_changes + right_changes + strip_changes

    @staticmethod
    def strip_collect_significant_changes(strip, threshold):
        """
        Find all significant price changes per day in the strip.
        """
        strip.sort(key=lambda point: point[1])
        significant_changes = []

        for i in range(len(strip)):
            for j in range(i + 1, min(i + 7, len(strip))):
                date1 = strip[i][0]
                date2 = strip[j][0]
                days_apart = abs((date2 - date1).days)
                if days_apart > 0:
                    change_per_day = abs(strip[j][1] - strip[i][1]) / days_apart
                    if change_per_day > threshold:
                        significant_changes.append((change_per_day, (strip[i], strip[j])))

        return significant_changes

    @staticmethod
    def brute_force(prices, threshold):
        """
        A brute-force method to find all significant price changes per day.
        """
        significant_changes = []

        for i in range(len(prices)):
            for j in range(i + 1, len(prices)):
                date1 = prices[i][0]
                date2 = prices[j][0]
                days_apart = abs((date2 - date1).days)
                if days_apart > 0:
                    change_per_day = abs(prices[i][1] - prices[j][1]) / days_apart
                    if change_per_day > threshold:
                        significant_changes.append((change_per_day, (prices[i], prices[j])))

        return significant_changes
    
# StockAnalyzer Class - Ties together data loading, sorting and analysis
class StockAnalyzer:
    def __init__(self, file_path):
        """
        Initialize the StockAnalyzer with a data loader for stock data.
        """
        self.loader = DataLoader(file_path)
        self.data = None

    def load_data(self):
        """
        Load the data using DataLoader.
        """
        self.data = self.loader.load_data()
        return self.data

    def sort_data(self, key_column):
        """
        Sort the data using the MergeSort class.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        sorter = MergeSort(key_column)
        sorted_data = sorter.sort(self.data)
        return sorted_data
    
    def find_max_gain_loss_period_1d(self):
        """
        Find the maximum gain/loss period in a 1D adjusted close price series
        """
        adj_close_column = self.data['Adj Close'].tolist()
        analyzer = MaxSubarray()
        max_gain = analyzer.find_max_gain(adj_close_column)
        return max_gain
    
    def find_max_gain_loss_period_2d(self):
        """
        Find the maximum gain/loss period in a 2D array of adjusted close prices.
        """
        adj_close_columns = [col for col in self.data.columns if 'Adj Close' in col]

        stock_prices = self.data[adj_close_columns].values  # 2D numpy array of stock prices
        analyzer = MaxSubarray()
        max_sum, top, bottom, left, right = analyzer.max_submatrix(stock_prices)
        return max_sum, top, bottom, left, right
        
    def detect_anomalies(self, threshold, min_days_apart=1):
        """
        Detect significant anomalies in the adjusted close prices.
        """
        anomalies = []
        adj_close_columns = [col for col in self.data.columns if 'Adj Close' in col]

        for col in adj_close_columns:
            prices = self.data[['Date', col]].values.tolist()
            
            significant_changes = SpikeDipClosestPair.find_all_significant_spikes_dips(prices, threshold)
            
            for change_per_day, closest_pair in significant_changes:
                date1 = closest_pair[0][0]
                date2 = closest_pair[1][0]
                days_apart = abs((date2 - date1).days)

                if days_apart >= min_days_apart:
                    anomalies.append((col, change_per_day, closest_pair))
                else:
                    print(f"Ignoring close pair in {col} due to insufficient days apart (Change: {change_per_day})")

        return anomalies

    def generate_max_gain_loss_report(self):
        """
        Generate a report for the period of maximum gain or loss, and plot the data.
        """
        if 'Date' not in self.data.columns:
            self.data.reset_index(inplace=True)

        adj_close_columns = [col for col in self.data.columns if 'Adj Close' in col]
        if len(adj_close_columns) == 1:
            adj_close_column = self.data['Adj Close'].tolist()
            dates = self.data['Date'].tolist()

            max_gain = self.find_max_gain_loss_period_1d()
            (max_gain_val, start_idx, end_idx) = max_gain
            
            plt.figure(figsize=(10, 6))
            plt.plot(dates, adj_close_column, label="Adj Close Price", color='blue')
            plt.axvspan(dates[start_idx], dates[end_idx], color='yellow', alpha=0.3, label='Max Gain/Loss Period')
            
            plt.title('Amazon Stock Prices with Maximum Gain/Loss Period Highlighted')
            plt.xlabel('Date')
            plt.ylabel('Adjusted Close Price')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            print(f"\nMax Gain Period: {dates[start_idx]} to {dates[end_idx]}")
            print(f"Maximum Gain Value: {max_gain_val}")
        else:
            max_gain = self.find_max_gain_loss_period_2d()
            
            max_sum, top, bottom, left, right = max_gain
            adj_close_columns = [col for col in self.data.columns if 'Adj Close' in col]
            
            dates = self.data['Date'].tolist()

            plt.figure(figsize=(10, 6))
            for col in adj_close_columns:
                plt.plot(dates, self.data[col], label=col)

            plt.axvspan(dates[top], dates[bottom], color='yellow', alpha=0.3, label='Max Gain/Loss Period')

            plt.title('Amazon Stock Prices with Maximum Gain/Loss Period Highlighted')
            plt.xlabel('Date')
            plt.ylabel('Adjusted Close Price')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            print(f"\nMaximum Gain/Loss (2D): {max_sum}")
            print(f"Submatrix Coordinates: Top={top}, Bottom={bottom}, Left={left}, Right={right}")

    def generate_trend_analysis_report(self):
        """
        Generate and plot the overall stock price trends.
        """
        self.data.set_index('Date', inplace=True)
        
        monthly_trend = self.data['Adj Close'].resample('ME').mean()
        yearly_trend = self.data['Adj Close'].resample('YE').mean()

        plt.figure(figsize=(10, 6))
        plt.plot(self.data.index, self.data['Adj Close'], label="Daily Adj Close Price", color='blue')
        plt.plot(monthly_trend.index, monthly_trend, label="Monthly Trend", color='green')
        plt.plot(yearly_trend.index, yearly_trend, label="Yearly Trend", color='red')
        
        plt.title('Amazon Stock Price Trends (Daily, Monthly, Yearly)')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
       
        print("Generated trend analysis report showing daily, monthly, and yearly trends.")

    def generate_anomaly_report(self, threshold=10):
        """
        Generate a report for detected anomalies and plot them on the stock price graph.
        """
        if 'Date' not in self.data.columns:
            self.data.reset_index(inplace=True)

        anomalies = self.detect_anomalies(threshold)
        
        plt.figure(figsize=(10, 6))
        dates = self.data['Date'].tolist()
        adj_close_column = self.data['Adj Close'].tolist()
        plt.plot(dates, adj_close_column, label="Adj Close Price", color='blue')

        for col, distance, pair in anomalies:
            date1 = pair[0][0]
            date2 = pair[1][0]
            price1 = pair[0][1]
            price2 = pair[1][1]
            plt.scatter([date1, date2], [price1, price2], color='red', label=f"Anomaly {col}", zorder=5)

        plt.title('Amazon Stock Prices with Detected Anomalies Highlighted')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        for col, distance, pair in anomalies:
            print(f"\nAnomalies detected for {col}:")
            print(f"Change per day: {distance}, Pair: {pair}")
    
def list_csv_files():
    """
    List all CSV files in the current directory.
    """
    return [f for f in os.listdir('stock_data') if f.endswith('.csv')]

def main():

    csv_files = list_csv_files()

    if not csv_files:
        print("No CSV files found in the current directory.")
        return

    print("Available CSV files:")
    for i, file in enumerate(csv_files):
        print(f"{i + 1}: {file}")

    while True:
        try:
            selected_files = input("Select one or more files (comma-separated numbers, e.g. 1,2): ")
            selected_indices = [int(i.strip()) - 1 for i in selected_files.split(',')]
            if any(i < 0 or i >= len(csv_files) for i in selected_indices):
                raise ValueError("Invalid file selection.")
            break
        except ValueError as e:
            print(f"Error: {e}. Please try again.")

    selected_files = [csv_files[i] for i in selected_indices]
    analyzer = StockAnalyzer(selected_files)
    analyzer.load_data()
    print("\nLoaded Data:")
    print(analyzer.data.head())

    available_reports = {
        1: "Max Gain/Loss Report",
        2: "Trend Analysis Report",
        3: "Anomaly Report"
    }

    if len(selected_files) > 1:
        print("You have selected multiple files. Only the Max Gain/Loss Report will be available.")
        available_reports = {1: "Max Gain/Loss Report"}

    print("Options for reports:")
    for key, report in available_reports.items():
        print(f"{key}: {report}")

    while True:
        try:
            if len(available_reports) > 1:
                report_choice = input("Select a report to generate (1, 2, or 3) or 'done' to finish: ")
            else:
                report_choice = input("Select a report to generate (1) or 'done' to finish: ")
            if report_choice.lower() == 'done':
                break
            report_choice = int(report_choice)
            if report_choice not in available_reports:
                raise ValueError("Invalid choice.")

            if report_choice == 1:
                analyzer.generate_max_gain_loss_report()
            elif report_choice == 2:
                analyzer.generate_trend_analysis_report()
            elif report_choice == 3:
                threshold = float(input("Enter the threshold for the anomaly report: "))
                analyzer.generate_anomaly_report(threshold)
        except ValueError as e:
            print(f"Error: {e}. Please try again.")

if __name__ == "__main__":
    main()