import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ====================== 1. LOAD DATA AND MODEL ======================
print("Loading data and model...")

try:
    # Load the new model
    model = joblib.load('work_new.joblib')
    print("✓ New model loaded successfully")
except FileNotFoundError:
    print("⚠️ 'work_new.joblib' not found, trying 'work.joblib'...")
    try:
        model = joblib.load('work.joblib')
        print("✓ Old model loaded")
    except:
        print("❌ No model file found")
        model = None

# Load the preprocessed data
try:
    # Try the new processed data first
    data = pd.read_csv('finaldata_processed.csv')
    print("✓ Loaded 'finaldata_processed.csv'")
except FileNotFoundError:
    try:
        data = pd.read_csv('finaldata.csv')
        print("✓ Loaded 'finaldata.csv'")
    except:
        print("❌ No data file found")
        data = None

if data is not None:
    # Drop the 'Unnamed' column if it exists
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    
    # Clean location names (remove 'Mumbai' suffix)
    if 'loc' in data.columns:
        data['loc'] = data['loc'].str.replace('Mumbai', '').str.strip()
        data['loc'] = data['loc'].str.replace('  ', ' ').str.strip()
    
    print(f"\nData shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"\nData summary:")
    print(data.describe().round(2))
    
    # ====================== 2. DATA OVERVIEW ======================
    print("\n" + "="*60)
    print("DATA OVERVIEW")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Mumbai Housing Data Distribution', fontsize=16, fontweight='bold')
    
    # Price distribution
    axes[0, 0].hist(data['price'] / 1000000, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Price (in Crores ₹)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Price Distribution')
    axes[0, 0].axvline(data['price'].mean() / 1000000, color='red', linestyle='--', 
                      label=f'Mean: ₹{data["price"].mean()/1000000:.1f}Cr')
    axes[0, 0].legend()
    
    # Size (BHK) distribution
    size_counts = data['size'].value_counts().sort_index()
    axes[0, 1].bar(size_counts.index, size_counts.values, alpha=0.7)
    axes[0, 1].set_xlabel('BHK')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('BHK Distribution')
    for i, v in enumerate(size_counts.values):
        axes[0, 1].text(size_counts.index[i], v, str(v), ha='center', va='bottom')
    
    # Area distribution
    axes[0, 2].hist(data['sqft'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 2].set_xlabel('Area (sq ft)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Area Distribution')
    axes[0, 2].axvline(data['sqft'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {data["sqft"].mean():.0f} sq ft')
    axes[0, 2].legend()
    
    # Bathroom distribution
    bath_counts = data['bath'].value_counts().sort_index()
    axes[1, 0].bar(bath_counts.index, bath_counts.values, alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Number of Bathrooms')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Bathroom Distribution')
    for i, v in enumerate(bath_counts.values):
        axes[1, 0].text(bath_counts.index[i], v, str(v), ha='center', va='bottom')
    
    # Parking distribution
    park_counts = data['parking'].value_counts().sort_index()
    axes[1, 1].bar(park_counts.index, park_counts.values, alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Parking Spots')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Parking Distribution')
    for i, v in enumerate(park_counts.values):
        axes[1, 1].text(park_counts.index[i], v, str(v), ha='center', va='bottom')
    
    # Furnishing distribution
    if 'furnishing' in data.columns:
        furnish_counts = data['furnishing'].value_counts()
        axes[1, 2].pie(furnish_counts.values, labels=furnish_counts.index, autopct='%1.1f%%',
                      startangle=90, colors=sns.color_palette("Set3"))
        axes[1, 2].set_title('Furnishing Status')
    
    plt.tight_layout()
    plt.show()
    
    # ====================== 3. CORRELATION ANALYSIS ======================
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Select only numeric columns for correlation
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        corr_matrix = data[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   mask=mask, square=True, linewidths=0.5, 
                   cbar_kws={"shrink": 0.8}, fmt='.2f')
        plt.title('Correlation Heatmap of Numeric Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print correlation with price
        if 'price' in corr_matrix.columns:
            print("\nCorrelation with Price:")
            price_corr = corr_matrix['price'].sort_values(ascending=False)
            for feature, corr_value in price_corr.items():
                if feature != 'price':
                    print(f"  {feature}: {corr_value:.3f}")
    
    # ====================== 4. LOCATION ANALYSIS ======================
    print("\n" + "="*60)
    print("LOCATION ANALYSIS")
    print("="*60)
    
    if 'loc' in data.columns:
        # Average prices by location
        avg_prices_by_loc = data.groupby('loc')['price'].agg(['mean', 'count', 'std']).round(2)
        avg_prices_by_loc = avg_prices_by_loc.sort_values('mean', ascending=False)
        
        print("\nTop 10 Most Expensive Locations:")
        print(avg_prices_by_loc.head(10).to_string())
        
        print("\nTop 10 Most Affordable Locations:")
        print(avg_prices_by_loc.tail(10).to_string())
        
        # Visualize top 15 locations
        top_locations = avg_prices_by_loc.head(15).index
        top_data = data[data['loc'].isin(top_locations)]
        
        plt.figure(figsize=(12, 6))
        order = avg_prices_by_loc.head(15).index
        sns.boxplot(x='loc', y='price', data=top_data, order=order)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Location')
        plt.ylabel('Price (₹)')
        plt.title('Price Distribution by Location (Top 15)', fontsize=14, fontweight='bold')
        
        # Format y-axis to show in crores
        ax = plt.gca()
        y_labels = [f'₹{int(y/1000000)}Cr' for y in ax.get_yticks()]
        ax.set_yticklabels(y_labels)
        
        plt.tight_layout()
        plt.show()
    
    # ====================== 5. BHK ANALYSIS ======================
    print("\n" + "="*60)
    print("BHK ANALYSIS")
    print("="*60)
    
    if 'size' in data.columns:
        avg_price_by_bhk = data.groupby('size').agg({
            'price': ['mean', 'count', 'min', 'max'],
            'sqft': 'mean'
        }).round(2)
        
        avg_price_by_bhk.columns = ['avg_price', 'count', 'min_price', 'max_price', 'avg_sqft']
        avg_price_by_bhk = avg_price_by_bhk.sort_index()
        
        print("\nPrice Statistics by BHK:")
        print(avg_price_by_bhk.to_string())
        
        # Visualize BHK vs Price
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar plot
        bars = ax1.bar(avg_price_by_bhk.index, avg_price_by_bhk['avg_price'] / 1000000, 
                      alpha=0.7, color=sns.color_palette("viridis", len(avg_price_by_bhk)))
        ax1.set_xlabel('BHK')
        ax1.set_ylabel('Average Price (Crores ₹)')
        ax1.set_title('Average Price by BHK')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bhk, row) in enumerate(avg_price_by_bhk.iterrows()):
            ax1.text(bhk, row['avg_price']/1000000 + 0.1, 
                    f'₹{row["avg_price"]/1000000:.1f}Cr\n({int(row["count"])})', 
                    ha='center', va='bottom', fontsize=9)
        
        # Scatter plot: BHK vs Price with area as size
        scatter = ax2.scatter(data['size'], data['price'] / 1000000, 
                             c=data['sqft'], alpha=0.6, cmap='viridis', s=50)
        ax2.set_xlabel('BHK')
        ax2.set_ylabel('Price (Crores ₹)')
        ax2.set_title('BHK vs Price (colored by Area)')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar for area
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Area (sq ft)')
        
        plt.tight_layout()
        plt.show()
    
    # ====================== 6. INTERACTIVE ANALYSIS ======================
    print("\n" + "="*60)
    print("INTERACTIVE ANALYSIS: BHK & LOCATION")
    print("="*60)
    
    if 'loc' in data.columns and 'size' in data.columns:
        # Create pivot table for heatmap
        pivot_data = data.pivot_table(
            values='price', 
            index='loc', 
            columns='size', 
            aggfunc='mean'
        ) / 1000000  # Convert to crores
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Price (Crores ₹)'})
        plt.title('Average Price by Location and BHK', fontsize=14, fontweight='bold')
        plt.xlabel('BHK')
        plt.ylabel('Location')
        plt.tight_layout()
        plt.show()
    
    # ====================== 7. AREA ANALYSIS ======================
    print("\n" + "="*60)
    print("AREA ANALYSIS")
    print("="*60)
    
    if 'sqft' in data.columns:
        # Price per sq ft
        data['price_per_sqft'] = data['price'] / data['sqft']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Scatter plot: Area vs Price
        scatter = axes[0].scatter(data['sqft'], data['price'] / 1000000, 
                                 alpha=0.6, s=20, c=data['size'], cmap='tab10')
        axes[0].set_xlabel('Area (sq ft)')
        axes[0].set_ylabel('Price (Crores ₹)')
        axes[0].set_title('Area vs Price (colored by BHK)')
        axes[0].grid(True, alpha=0.3)
        
        # Add colorbar for BHK
        cbar = plt.colorbar(scatter, ax=axes[0])
        cbar.set_label('BHK')
        
        # Price per sq ft distribution
        axes[1].hist(data['price_per_sqft'], bins=40, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Price per sq ft (₹)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Price per sq ft')
        axes[1].axvline(data['price_per_sqft'].mean(), color='red', linestyle='--',
                       label=f'Mean: ₹{data["price_per_sqft"].mean():.0f}/sq ft')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nAverage price per sq ft: ₹{data['price_per_sqft'].mean():.0f}")
        print(f"Minimum price per sq ft: ₹{data['price_per_sqft'].min():.0f}")
        print(f"Maximum price per sq ft: ₹{data['price_per_sqft'].max():.0f}")
    
    # ====================== 8. MODEL INSIGHTS ======================
    print("\n" + "="*60)
    print("MODEL INSIGHTS")
    print("="*60)
    
    if model is not None and hasattr(model, 'named_steps'):
        try:
            # Get feature importance if available
            if hasattr(model.named_steps['ridge'], 'coef_'):
                coefficients = model.named_steps['ridge'].coef_
            elif hasattr(model.named_steps['linearregression'], 'coef_'):
                coefficients = model.named_steps['linearregression'].coef_
            else:
                coefficients = None
            
            if coefficients is not None:
                # Get feature names from the pipeline
                col_transformer = model.named_steps['columntransformer']
                ohe = col_transformer.named_transformers_['onehotencoder']
                
                # Get categorical feature names
                cat_features = ohe.get_feature_names_out(['loc', 'furnishing'])
                num_features = ['sqft', 'size', 'bath', 'parking']
                all_features = list(cat_features) + num_features
                
                # Create feature importance DataFrame
                feature_importance = pd.DataFrame({
                    'feature': all_features,
                    'coefficient': coefficients
                })
                
                # Sort by absolute value
                feature_importance['abs_coef'] = np.abs(feature_importance['coefficient'])
                feature_importance = feature_importance.sort_values('abs_coef', ascending=False)
                
                print("\nTop 20 Most Important Features:")
                print(feature_importance.head(20).to_string(index=False))
                
                # Plot top features
                top_features = feature_importance.head(15)
                
                plt.figure(figsize=(12, 8))
                colors = ['red' if coef < 0 else 'green' for coef in top_features['coefficient']]
                bars = plt.barh(top_features['feature'], top_features['coefficient'], color=colors)
                plt.xlabel('Coefficient Value (Impact on Price)')
                plt.title('Top 15 Features Affecting House Price', fontsize=14, fontweight='bold')
                plt.gca().invert_yaxis()  # Highest at top
                plt.grid(True, alpha=0.3, axis='x')
                
                # Add coefficient values on bars
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width, bar.get_y() + bar.get_height()/2, 
                            f'{width:.2f}', ha='left' if width >= 0 else 'right', 
                            va='center', fontsize=9)
                
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            print(f"Could not extract model coefficients: {e}")
    
    # ====================== 9. SUMMARY STATISTICS ======================
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    summary_stats = {
        'Total Properties': len(data),
        'Average Price': f'₹{data["price"].mean():,.0f}',
        'Median Price': f'₹{data["price"].median():,.0f}',
        'Price Range': f'₹{data["price"].min():,.0f} - ₹{data["price"].max():,.0f}',
        'Avg Area': f'{data["sqft"].mean():.0f} sq ft',
        'Most Common BHK': int(data['size'].mode()[0]),
        'Avg BHK': f'{data["size"].mean():.1f}',
        'Avg Bathrooms': f'{data["bath"].mean():.1f}',
        'Avg Parking': f'{data["parking"].mean():.1f}',
    }
    
    for key, value in summary_stats.items():
        print(f"{key:20}: {value}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

else:
    print("No data available for analysis.")