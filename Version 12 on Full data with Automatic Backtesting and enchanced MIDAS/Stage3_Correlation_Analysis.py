import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from datetime import datetime

def generate_comprehensive_correlation_heatmap(data_file_path, output_folder='Correlation_Analysis'):
    """
    Generate comprehensive correlation heatmaps for economic time series data
    
    Parameters:
    -----------
    data_file_path : str
        Path to the processed data CSV file
    output_folder : str
        Folder to save the correlation analysis outputs
    """
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Load the processed data
        df = pd.read_csv(data_file_path, index_col=0, parse_dates=True)
        logging.info(f"Loaded data with shape: {df.shape}")
        logging.info(f"Variables: {list(df.columns)}")
        
        # Create output directory
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Calculate basic statistics for each variable
        stats_summary = pd.DataFrame({
            'Variable': df.columns,
            'Total_Observations': [len(df) for _ in df.columns],
            'Valid_Observations': [df[col].notna().sum() for col in df.columns],
            'Coverage_Percent': [round((df[col].notna().sum() / len(df)) * 100, 1) for col in df.columns],
            'Mean': [round(df[col].mean(), 4) for col in df.columns],
            'Std': [round(df[col].std(), 4) for col in df.columns],
            'Min': [round(df[col].min(), 4) for col in df.columns],
            'Max': [round(df[col].max(), 4) for col in df.columns]
        })
        
        # Save summary statistics
        stats_file = os.path.join(output_folder, 'Variable_Summary_Statistics.csv')
        stats_summary.to_csv(stats_file, index=False)
        logging.info(f"Saved summary statistics to '{stats_file}'")
        
        # 1. Full Correlation Matrix (all variables)
        logging.info("Generating full correlation matrix...")
        corr_matrix_full = df.corr()
        
        # Create full heatmap
        plt.figure(figsize=(max(12, len(df.columns) * 0.8), max(10, len(df.columns) * 0.7)))
        mask = np.triu(np.ones_like(corr_matrix_full, dtype=bool))  # Mask upper triangle
        
        sns.heatmap(
            corr_matrix_full,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
            annot_kws={'size': max(6, 12 - len(df.columns) * 0.15)}
        )
        
        plt.title(f'Correlation Heatmap: All Economic Indicators\n'
                  f'({len(df.columns)} variables, {len(df)} time periods)\n'
                  f'Data Range: {df.index.min().strftime("%Y-%m")} to {df.index.max().strftime("%Y-%m")}',
                  fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        full_heatmap_file = os.path.join(output_folder, 'Full_Correlation_Heatmap.png')
        plt.savefig(full_heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved full correlation heatmap to '{full_heatmap_file}'")
        
        # 2. High Coverage Variables Only (>= 50% data availability)
        high_coverage_vars = [col for col in df.columns if df[col].notna().sum() >= len(df) * 0.5]
        
        if len(high_coverage_vars) > 1:
            logging.info(f"Generating heatmap for {len(high_coverage_vars)} high-coverage variables...")
            
            corr_matrix_high = df[high_coverage_vars].corr()
            
            plt.figure(figsize=(max(10, len(high_coverage_vars) * 0.9), max(8, len(high_coverage_vars) * 0.8)))
            sns.heatmap(
                corr_matrix_high,
                annot=True,
                fmt='.3f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.8,
                cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
                annot_kws={'size': max(8, 14 - len(high_coverage_vars) * 0.2)}
            )
            
            plt.title(f'Correlation Heatmap: High Coverage Variables (≥50% data)\n'
                      f'({len(high_coverage_vars)} variables)',
                      fontsize=14, pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            high_coverage_file = os.path.join(output_folder, 'High_Coverage_Correlation_Heatmap.png')
            plt.savefig(high_coverage_file, dpi=300, bbox_inches='tight')
            plt.close()
            logging.info(f"Saved high-coverage heatmap to '{high_coverage_file}'")
        
        # 3. Clustered Correlation Heatmap (reorder variables by similarity)
        logging.info("Generating clustered correlation heatmap...")
        
        # Use hierarchical clustering to reorder variables
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        
        # Convert correlation to distance matrix
        distance_matrix = 1 - abs(corr_matrix_full)
        condensed_distances = squareform(distance_matrix, checks=False)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_distances, method='average')
        
        # Create clustered heatmap
        g = sns.clustermap(
            corr_matrix_full,
            method='average',
            metric='correlation',
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.2f',
            figsize=(max(12, len(df.columns) * 0.8), max(10, len(df.columns) * 0.7)),
            annot_kws={'size': max(6, 12 - len(df.columns) * 0.15)},
            cbar_kws={"shrink": .8, "label": "Correlation Coefficient"}
        )
        
        g.fig.suptitle(f'Clustered Correlation Heatmap: Economic Indicators\n'
                       f'Variables grouped by correlation similarity',
                       fontsize=14, y=0.98)
        
        clustered_file = os.path.join(output_folder, 'Clustered_Correlation_Heatmap.png')
        g.savefig(clustered_file, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved clustered heatmap to '{clustered_file}'")
        
        # 4. Generate correlation statistics table
        logging.info("Generating correlation statistics...")
        
        # Flatten correlation matrix and get statistics
        corr_values = corr_matrix_full.values
        mask = np.triu(np.ones_like(corr_values, dtype=bool), k=1)  # Upper triangle excluding diagonal
        upper_triangle = corr_values[mask]
        
        correlation_stats = {
            'Total_Pairs': len(upper_triangle),
            'Mean_Correlation': round(np.mean(upper_triangle), 4),
            'Median_Correlation': round(np.median(upper_triangle), 4),
            'Std_Correlation': round(np.std(upper_triangle), 4),
            'Min_Correlation': round(np.min(upper_triangle), 4),
            'Max_Correlation': round(np.max(upper_triangle), 4),
            'Strong_Positive_Corr_Count': int(np.sum(upper_triangle >= 0.7)),
            'Moderate_Positive_Corr_Count': int(np.sum((upper_triangle >= 0.3) & (upper_triangle < 0.7))),
            'Weak_Corr_Count': int(np.sum((upper_triangle >= -0.3) & (upper_triangle < 0.3))),
            'Moderate_Negative_Corr_Count': int(np.sum((upper_triangle >= -0.7) & (upper_triangle < -0.3))),
            'Strong_Negative_Corr_Count': int(np.sum(upper_triangle < -0.7))
        }
        
        # Save correlation statistics
        corr_stats_df = pd.DataFrame([correlation_stats])
        corr_stats_file = os.path.join(output_folder, 'Correlation_Statistics.csv')
        corr_stats_df.to_csv(corr_stats_file, index=False)
        
        # 5. Top correlations table
        corr_pairs = []
        n_vars = len(df.columns)
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                var1, var2 = df.columns[i], df.columns[j]
                corr_val = corr_matrix_full.iloc[i, j]
                if not np.isnan(corr_val):
                    corr_pairs.append({
                        'Variable_1': var1,
                        'Variable_2': var2,
                        'Correlation': round(corr_val, 4),
                        'Abs_Correlation': round(abs(corr_val), 4)
                    })
        
        # Sort by absolute correlation value
        corr_pairs_df = pd.DataFrame(corr_pairs)
        corr_pairs_df = corr_pairs_df.sort_values('Abs_Correlation', ascending=False)
        
        # Save top 20 correlations
        top_corr_file = os.path.join(output_folder, 'Top_20_Correlations.csv')
        corr_pairs_df.head(20).to_csv(top_corr_file, index=False)
        logging.info(f"Saved top correlations to '{top_corr_file}'")
        
        # 6. Save full correlation matrix
        corr_matrix_file = os.path.join(output_folder, 'Full_Correlation_Matrix.csv')
        corr_matrix_full.to_csv(corr_matrix_file)
        logging.info(f"Saved correlation matrix to '{corr_matrix_file}'")
        
        # Print summary
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total variables analyzed: {len(df.columns)}")
        print(f"Time period: {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")
        print(f"Total observations per variable: {len(df)}")
        print(f"\nCorrelation Statistics:")
        print(f"  Mean correlation: {correlation_stats['Mean_Correlation']}")
        print(f"  Strongest positive: {correlation_stats['Max_Correlation']}")
        print(f"  Strongest negative: {correlation_stats['Min_Correlation']}")
        print(f"  Strong correlations (|r| ≥ 0.7): {correlation_stats['Strong_Positive_Corr_Count'] + correlation_stats['Strong_Negative_Corr_Count']}")
        print(f"\nFiles generated in '{output_folder}':")
        print(f"  - Full_Correlation_Heatmap.png")
        print(f"  - High_Coverage_Correlation_Heatmap.png")
        print(f"  - Clustered_Correlation_Heatmap.png")
        print(f"  - Full_Correlation_Matrix.csv")
        print(f"  - Top_20_Correlations.csv")
        print(f"  - Correlation_Statistics.csv")
        print(f"  - Variable_Summary_Statistics.csv")
        
        return corr_matrix_full
        
    except FileNotFoundError:
        logging.error(f"Data file '{data_file_path}' not found.")
        return None
    except Exception as e:
        logging.error(f"Error generating correlation analysis: {str(e)}")
        return None


def main():
    """
    Main function to run correlation analysis
    Modify the file paths as needed for your data
    """
    
    # Update these paths to match your data location
    processed_data_file = 'Processed Data/All_Processed_data.csv'
    
    # Alternative: if you want to use the raw data file
    # raw_data_file = 'Sample Clipped Data.csv'
    
    print("Starting comprehensive correlation analysis...")
    
    if os.path.exists(processed_data_file):
        correlation_matrix = generate_comprehensive_correlation_heatmap(processed_data_file)
    else:
        print(f"Processed data file not found: {processed_data_file}")
        print("Please run your preprocessing pipeline first, or update the file path.")
        
        # Alternative: analyze raw data directly if available
        raw_data_file = 'Sample Clipped Data.csv'
        if os.path.exists(raw_data_file):
            print(f"Attempting to analyze raw data from: {raw_data_file}")
            correlation_matrix = generate_comprehensive_correlation_heatmap(raw_data_file)
        else:
            print(f"Raw data file also not found: {raw_data_file}")
            print("Please ensure your data files are in the correct location.")


if __name__ == '__main__':
    main()