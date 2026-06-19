"""
Telecom Investment Intelligence Platform - Business Feature Engineering
Creates 35 business-focused engineered features for executive decision-making
Prioritizes network infrastructure investments, revenue protection, and customer value
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Tuple, Dict

warnings.filterwarnings('ignore')

class BusinessFeatureEngineer:
    """
    Business-focused feature engineering for telecom investment intelligence.
    Creates features that drive executive decision-making rather than just model accuracy.
    """
    
    def __init__(self, client_path='Client.csv', record_path='Record.csv'):
        self.client_path = client_path
        self.record_path = record_path
        self.client_df = None
        self.record_df = None
        self.merged_df = None
        self.engineered_df = None
        self.area_features = None
        self.output_dir = Path('feature_engineering_output')
        self.output_dir.mkdir(exist_ok=True)
        
    def load_data(self) -> bool:
        """Load and merge the telecom datasets."""
        print("=" * 70)
        print("LOADING DATASETS FOR FEATURE ENGINEERING")
        print("=" * 70)
        
        try:
            self.client_df = pd.read_csv(self.client_path)
            print(f"[OK] Client.csv loaded: {self.client_df.shape[0]} rows, {self.client_df.shape[1]} columns")
        except Exception as e:
            print(f"[ERROR] Error loading Client.csv: {e}")
            return False
            
        try:
            self.record_df = pd.read_csv(self.record_path)
            print(f"[OK] Record.csv loaded: {self.record_df.shape[0]} rows, {self.record_df.shape[1]} columns")
        except Exception as e:
            print(f"[ERROR] Error loading Record.csv: {e}")
            return False
        
        # Merge datasets on Customer_ID
        try:
            self.merged_df = pd.merge(
                self.client_df, 
                self.record_df, 
                on='Customer_ID', 
                how='inner',
                suffixes=('_client', '_record')
            )
            print(f"[OK] Datasets merged: {self.merged_df.shape[0]} rows, {self.merged_df.shape[1]} columns")
            return True
        except Exception as e:
            print(f"[ERROR] Error merging datasets: {e}")
            return False
    
    def safe_divide(self, numerator, denominator, default=0):
        """Safe division that handles division by zero."""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(numerator, denominator)
            result[~np.isfinite(result)] = default
        return result
    
    def create_network_quality_features(self):
        """Create Network Quality features (1-8) for infrastructure investment decisions."""
        print("\n" + "=" * 70)
        print("CREATING NETWORK QUALITY FEATURES")
        print("=" * 70)
        
        df = self.merged_df.copy()
        
        # Feature 1: Network_Health_Index
        # Overall network performance score (0-100) combining drops and blocks
        df['Network_Health_Index'] = 100 - (
            (df['drop_vce_Mean'].fillna(0) + df['drop_dat_Mean'].fillna(0)) * 10 +
            (df['blck_vce_Mean'].fillna(0) + df['blck_dat_Mean'].fillna(0)) * 5
        )
        df['Network_Health_Index'] = df['Network_Health_Index'].clip(0, 100)
        print("[OK] Feature 1: Network_Health_Index created")
        
        # Feature 2: Voice_Quality_Score
        # Voice-specific network quality for revenue-generating calls
        df['Voice_Quality_Score'] = 100 - (
            df['drop_vce_Mean'].fillna(0) * 15 + 
            df['blck_vce_Mean'].fillna(0) * 8
        )
        df['Voice_Quality_Score'] = df['Voice_Quality_Score'].clip(0, 100)
        print("[OK] Feature 2: Voice_Quality_Score created")
        
        # Feature 3: Data_Quality_Score
        # Data-specific network quality for high-margin data services
        df['Data_Quality_Score'] = 100 - (
            df['drop_dat_Mean'].fillna(0) * 15 + 
            df['blck_dat_Mean'].fillna(0) * 8
        )
        df['Data_Quality_Score'] = df['Data_Quality_Score'].clip(0, 100)
        print("[OK] Feature 3: Data_Quality_Score created")
        
        # Feature 4: Call_Completion_Rate
        # Percentage of successfully completed calls
        df['Call_Completion_Rate'] = self.safe_divide(
            df['complete_Mean'].fillna(0),
            df['attempt_Mean'].fillna(1),
            default=0
        ) * 100
        df['Call_Completion_Rate'] = df['Call_Completion_Rate'].clip(0, 100)
        print("[OK] Feature 4: Call_Completion_Rate created")
        
        # Feature 5: Network_Risk_Level
        # Categorical risk assessment for network investments
        def categorize_network_risk(health_index):
            if health_index < 70:
                return 'CRITICAL'
            elif health_index < 85:
                return 'WARNING'
            else:
                return 'HEALTHY'
        
        df['Network_Risk_Level'] = df['Network_Health_Index'].apply(categorize_network_risk)
        print("[OK] Feature 5: Network_Risk_Level created")
        
        # Feature 6: Peak_Hour_Quality
        # Network quality during high-demand periods
        peak_usage = df['mou_peav_Mean'].fillna(0) + df['mou_pead_Mean'].fillna(0)
        peak_quality = df['peak_vce_Mean'].fillna(0) + df['peak_dat_Mean'].fillna(0)
        df['Peak_Hour_Quality'] = self.safe_divide(peak_quality, peak_usage, default=0)
        print("[OK] Feature 6: Peak_Hour_Quality created")
        
        # Feature 7: Service_Failure_Impact
        # Revenue at risk due to network failures
        total_failures = df['drop_vce_Mean'].fillna(0) + df['drop_dat_Mean'].fillna(0)
        df['Service_Failure_Impact'] = total_failures * df['totrev'].fillna(0)
        print("[OK] Feature 7: Service_Failure_Impact created")
        
        # Feature 8: Customer_Support_Load_Index
        # Support calls per 1000 minutes of usage
        df['Customer_Support_Load_Index'] = self.safe_divide(
            df['custcare_Mean'].fillna(0),
            df['totmou'].fillna(1) / 1000,
            default=0
        )
        print("[OK] Feature 8: Customer_Support_Load_Index created")
        
        self.merged_df = df
        return df
    
    def create_revenue_risk_features(self):
        """Create Revenue Risk features (9-14) for revenue protection."""
        print("\n" + "=" * 70)
        print("CREATING REVENUE RISK FEATURES")
        print("=" * 70)
        
        df = self.merged_df.copy()
        
        # Feature 9: Revenue_Stability_Score
        # Consistency of revenue over time (0-1)
        revenue_cols = ['avg3rev', 'avg6rev', 'rev_Mean']
        available_rev_cols = [col for col in revenue_cols if col in df.columns]
        
        if len(available_rev_cols) >= 2:
            rev_data = df[available_rev_cols].fillna(0)
            rev_std = rev_data.std(axis=1)
            rev_mean = rev_data.mean(axis=1)
            df['Revenue_Stability_Score'] = 1 - self.safe_divide(rev_std, rev_mean, default=1)
            df['Revenue_Stability_Score'] = df['Revenue_Stability_Score'].clip(0, 1)
        else:
            df['Revenue_Stability_Score'] = 0.5  # Default if insufficient data
        print("[OK] Feature 9: Revenue_Stability_Score created")
        
        # Feature 10: Revenue_At_Risk_Flag
        # Customers with declining revenue AND poor network quality
        declining_revenue = df['change_rev'].fillna(0) < -20
        poor_network = df['Network_Health_Index'] < 75
        df['Revenue_At_Risk_Flag'] = (declining_revenue & poor_network).astype(int)
        print("[OK] Feature 10: Revenue_At_Risk_Flag created")
        
        # Feature 11: Usage_Revenue_Efficiency
        # Revenue generated per minute of usage
        df['Usage_Revenue_Efficiency'] = self.safe_divide(
            df['totrev'].fillna(0),
            df['totmou'].fillna(1),
            default=0
        )
        print("[OK] Feature 11: Usage_Revenue_Efficiency created")
        
        # Feature 12: Revenue_Vulnerability_Index
        # Combines revenue trends with network quality risks
        revenue_change = self.safe_divide(
            df['rev_Mean'].fillna(0) - df['avg3rev'].fillna(0),
            df['rev_Mean'].fillna(1),
            default=0
        )
        network_risk_factor = 1 - (df['Network_Health_Index'] / 100)
        df['Revenue_Vulnerability_Index'] = revenue_change * network_risk_factor
        print("[OK] Feature 12: Revenue_Vulnerability_Index created")
        
        # Feature 13: Premium_Revenue_Ratio
        # Percentage of revenue from overage (premium pricing)
        df['Premium_Revenue_Ratio'] = self.safe_divide(
            df['ovrrev_Mean'].fillna(0),
            df['rev_Mean'].fillna(1),
            default=0
        )
        df['Premium_Revenue_Ratio'] = df['Premium_Revenue_Ratio'].clip(0, 1)
        print("[OK] Feature 13: Premium_Revenue_Ratio created")
        
        # Feature 14: Revenue_Concentration_Risk
        # Individual customer revenue contribution to total area revenue
        # This will be calculated at area level later
        df['Revenue_Concentration_Risk'] = df['rev_Mean'].fillna(0) / df['totrev'].fillna(1)
        print("[OK] Feature 14: Revenue_Concentration_Risk created")
        
        self.merged_df = df
        return df
    
    def create_customer_value_features(self):
        """Create Customer Value features (15-20) for segmentation and retention."""
        print("\n" + "=" * 70)
        print("CREATING CUSTOMER VALUE FEATURES")
        print("=" * 70)
        
        df = self.merged_df.copy()
        
        # Feature 15: Customer_Lifetime_Value_Score
        # Estimated lifetime value combining revenue, tenure, and equipment investment
        equipment_factor = 1 + (df['hnd_price'].fillna(0) / 100)
        tenure_years = df['months'].fillna(0) / 12
        df['Customer_Lifetime_Value_Score'] = (
            df['rev_Mean'].fillna(0) * tenure_years * equipment_factor
        )
        print("[OK] Feature 15: Customer_Lifetime_Value_Score created")
        
        # Feature 16: Equipment_Investment_Index
        # Equipment value adjusted for age (depreciates over 10 years)
        depreciation_factor = 1 - self.safe_divide(df['eqpdays'].fillna(0), 3650, default=0)
        df['Equipment_Investment_Index'] = df['hnd_price'].fillna(0) * depreciation_factor.clip(0, 1)
        print("[OK] Feature 16: Equipment_Investment_Index created")
        
        # Feature 17: Subscription_Utilization_Rate
        # Active subscriptions as percentage of total subscriptions
        df['Subscription_Utilization_Rate'] = self.safe_divide(
            df['actvsubs'].fillna(0),
            df['uniqsubs'].fillna(1),
            default=0
        )
        df['Subscription_Utilization_Rate'] = df['Subscription_Utilization_Rate'].clip(0, 1)
        print("[OK] Feature 17: Subscription_Utilization_Rate created")
        
        # Feature 18: Revenue_Per_Subscription
        # Average revenue generated per subscription
        df['Revenue_Per_Subscription'] = self.safe_divide(
            df['totrev'].fillna(0),
            df['uniqsubs'].fillna(1),
            default=0
        )
        print("[OK] Feature 18: Revenue_Per_Subscription created")
        
        # Feature 19: Customer_Engagement_Score
        # Engagement combining usage, call completion, and device count
        monthly_usage = self.safe_divide(df['totmou'].fillna(0), df['months'].fillna(1), default=0)
        completion_factor = df['Call_Completion_Rate'] / 100
        device_factor = 1 + (df['phones'].fillna(0) / 10)
        df['Customer_Engagement_Score'] = monthly_usage * completion_factor * device_factor
        print("[OK] Feature 19: Customer_Engagement_Score created")
        
        # Feature 20: Credit_Quality_Adjusted_Value
        # Customer value adjusted for credit quality
        credit_multiplier = np.where(df['creditcd'].fillna('N') == 'Y', 1.2, 0.8)
        df['Credit_Quality_Adjusted_Value'] = df['Customer_Lifetime_Value_Score'] * credit_multiplier
        print("[OK] Feature 20: Credit_Quality_Adjusted_Value created")
        
        self.merged_df = df
        return df
    
    def create_growth_potential_features(self):
        """Create Growth Potential features (21-26) for upsell and revenue growth."""
        print("\n" + "=" * 70)
        print("CREATING GROWTH POTENTIAL FEATURES")
        print("=" * 70)
        
        df = self.merged_df.copy()
        
        # Feature 21: Usage_Growth_Trend
        # Percentage change in usage from 3-month to 6-month average
        df['Usage_Growth_Trend'] = self.safe_divide(
            df['avg6mou'].fillna(0) - df['avg3mou'].fillna(0),
            df['avg3mou'].fillna(1),
            default=0
        )
        print("[OK] Feature 21: Usage_Growth_Trend created")
        
        # Feature 22: Revenue_Growth_Momentum
        # Revenue growth acceleration metric
        df['Revenue_Growth_Momentum'] = self.safe_divide(
            df['avg6rev'].fillna(0) - df['avg3rev'].fillna(0),
            df['avg3rev'].fillna(1),
            default=0
        )
        print("[OK] Feature 22: Revenue_Growth_Momentum created")
        
        # Feature 23: Data_Adoption_Index
        # Data usage as percentage of total usage
        total_usage = df['da_Mean'].fillna(0) + df['mou_Mean'].fillna(0)
        df['Data_Adoption_Index'] = self.safe_divide(df['da_Mean'].fillna(0), total_usage, default=0)
        df['Data_Adoption_Index'] = df['Data_Adoption_Index'].clip(0, 1)
        print("[OK] Feature 23: Data_Adoption_Index created")
        
        # Feature 24: Upsell_Potential_Score
        # Composite score identifying customers ready for premium services
        growth_component = df['Usage_Growth_Trend'] * 0.4
        data_component = df['Data_Adoption_Index'] * 0.3
        equipment_component = (df['hnd_price'].fillna(0) / 200) * 0.3
        df['Upsell_Potential_Score'] = growth_component + data_component + equipment_component
        print("[OK] Feature 24: Upsell_Potential_Score created")
        
        # Feature 25: Cross_Sell_Opportunity
        # Revenue potential from activating inactive subscriptions
        inactive_subs = (df['uniqsubs'].fillna(0) - df['actvsubs'].fillna(0)).clip(0)
        df['Cross_Sell_Opportunity'] = np.where(
            inactive_subs > 0,
            inactive_subs * df['rev_Mean'].fillna(0),
            0
        )
        print("[OK] Feature 25: Cross_Sell_Opportunity created")
        
        # Feature 26: Tenure_Based_Growth_Potential
        # Growth potential adjusted for customer lifecycle stage
        def tenure_multiplier(months):
            if months < 12:
                return 1.5
            elif months < 24:
                return 1.2
            else:
                return 1.0
        
        df['Tenure_Based_Growth_Potential'] = (
            df['months'].fillna(0).apply(tenure_multiplier) * df['Usage_Growth_Trend']
        )
        print("[OK] Feature 26: Tenure_Based_Growth_Potential created")
        
        self.merged_df = df
        return df
    
    def create_infrastructure_investment_features(self):
        """Create Infrastructure Investment features (27-31) for capital allocation."""
        print("\n" + "=" * 70)
        print("CREATING INFRASTRUCTURE INVESTMENT FEATURES")
        print("=" * 70)
        
        df = self.merged_df.copy()
        
        # Feature 27: Infrastructure_Priority_Score
        # Priority score for infrastructure investments (customer-level)
        # This will be aggregated at area level later
        df['Infrastructure_Priority_Base'] = (
            df['Network_Health_Index'] * df['rev_Mean'].fillna(0)
        ) / 1000
        print("[OK] Feature 27: Infrastructure_Priority_Base created (will aggregate at area level)")
        
        # Feature 28: Capacity_Utilization_Rate
        # Peak usage as percentage of average daily usage
        avg_daily_usage = self.safe_divide(df['totmou'].fillna(0), 30, default=1)
        peak_usage = df['peak_vce_Mean'].fillna(0) + df['peak_dat_Mean'].fillna(0)
        df['Capacity_Utilization_Rate'] = self.safe_divide(peak_usage, avg_daily_usage, default=0)
        print("[OK] Feature 28: Capacity_Utilization_Rate created")
        
        # Feature 29: Revenue_At_Risk_Per_Tower
        # Financial justification for specific tower investments
        # Will be refined with actual tower data
        df['Revenue_At_Risk_Amount'] = df['Revenue_At_Risk_Flag'] * df['rev_Mean'].fillna(0)
        print("[OK] Feature 29: Revenue_At_Risk_Amount created (per-tower requires tower mapping)")
        
        # Feature 30: Support_Cost_Driver_Index
        # Customer support costs driven by network quality issues
        network_quality_factor = 1 - (df['Network_Health_Index'] / 100)
        df['Support_Cost_Driver_Index'] = df['custcare_Mean'].fillna(0) * network_quality_factor
        print("[OK] Feature 30: Support_Cost_Driver_Index created")
        
        # Feature 31: Roaming_Revenue_Opportunity
        # Revenue potential from roaming services
        df['Roaming_Revenue_Opportunity'] = self.safe_divide(
            df['roam_Mean'].fillna(0) * df['rev_Mean'].fillna(0),
            df['totmou'].fillna(1),
            default=0
        )
        print("[OK] Feature 31: Roaming_Revenue_Opportunity created")
        
        self.merged_df = df
        return df
    
    def create_area_level_features(self):
        """Create Area-level aggregation features (32-35) for regional investment decisions."""
        print("\n" + "=" * 70)
        print("CREATING AREA-LEVEL AGGREGATION FEATURES")
        print("=" * 70)
        
        df = self.merged_df.copy()
        
        # Feature 32: Area_Investment_Priority_Score (calculated after all area features)
        # This will be calculated at the end
        
        # Feature 33: Area_Network_Health_Index
        area_network_health = df.groupby('area')['Network_Health_Index'].mean().reset_index()
        area_network_health.columns = ['area', 'Area_Network_Health_Index']
        df = df.merge(area_network_health, on='area', how='left')
        print("[OK] Feature 33: Area_Network_Health_Index created")
        
        # Feature 34: Area_Revenue_Per_Customer
        area_revenue = df.groupby('area')['totrev'].sum().reset_index()
        area_customers = df.groupby('area')['Customer_ID'].count().reset_index()
        area_revenue_per_customer = pd.merge(
            area_revenue, 
            area_customers, 
            on='area'
        )
        area_revenue_per_customer['Area_Revenue_Per_Customer'] = self.safe_divide(
            area_revenue_per_customer['totrev'],
            area_revenue_per_customer['Customer_ID'],
            default=0
        )
        area_revenue_per_customer = area_revenue_per_customer[['area', 'Area_Revenue_Per_Customer']]
        df = df.merge(area_revenue_per_customer, on='area', how='left')
        print("[OK] Feature 34: Area_Revenue_Per_Customer created")
        
        # Feature 35: Area_Customer_Density
        # Customer concentration by geographic area (assuming equal area size for now)
        area_customer_count = df.groupby('area')['Customer_ID'].count().reset_index()
        area_customer_count.columns = ['area', 'Area_Customer_Count']
        df = df.merge(area_customer_count, on='area', how='left')
        
        # Since we don't have actual area square miles, we'll use relative density
        df['Area_Customer_Density'] = df['Area_Customer_Count']  # Can be normalized with actual area data
        print("[OK] Feature 35: Area_Customer_Density created")
        
        # Now calculate Feature 32: Area_Investment_Priority_Score
        # Additional area-level aggregations needed
        area_growth_rate = df.groupby('area')['Revenue_Growth_Momentum'].mean().reset_index()
        area_growth_rate.columns = ['area', 'Area_Revenue_Growth_Rate']
        
        area_risk_amount = df.groupby('area')['Revenue_At_Risk_Amount'].sum().reset_index()
        area_risk_amount.columns = ['area', 'Area_Revenue_At_Risk']
        
        area_capacity_util = df.groupby('area')['Capacity_Utilization_Rate'].mean().reset_index()
        area_capacity_util.columns = ['area', 'Area_Capacity_Utilization']
        
        area_upsell_potential = df.groupby('area')['Upsell_Potential_Score'].mean().reset_index()
        area_upsell_potential.columns = ['area', 'Area_Customer_Growth_Potential']
        
        # Merge all area-level features
        area_features = area_customer_count.merge(area_revenue_per_customer, on='area')
        area_features = area_features.merge(area_network_health, on='area')
        area_features = area_features.merge(area_growth_rate, on='area')
        area_features = area_features.merge(area_risk_amount, on='area')
        area_features = area_features.merge(area_capacity_util, on='area')
        area_features = area_features.merge(area_upsell_potential, on='area')
        
        # Calculate Area Investment Priority Score
        # Formula: (Customer_Count * 0.25) + (Revenue_Per_Customer * 0.20) + 
        # ((100 - Network_Health) * 0.20) + (Revenue_At_Risk * 0.15) + 
        # (Capacity_Utilization * 0.10) + (Growth_Potential * 0.10)
        
        area_features['Area_Investment_Priority_Score'] = (
            (area_features['Area_Customer_Count'] * 0.25) +
            (area_features['Area_Revenue_Per_Customer'] * 0.20) +
            ((100 - area_features['Area_Network_Health_Index']) * 0.20) +
            (area_features['Area_Revenue_At_Risk'] * 0.15) +
            (area_features['Area_Capacity_Utilization'] * 0.10) +
            (area_features['Area_Customer_Growth_Potential'] * 0.10)
        )
        
        # Merge back to main dataframe
        area_priority = area_features[['area', 'Area_Investment_Priority_Score']]
        df = df.merge(area_priority, on='area', how='left')
        
        print("[OK] Feature 32: Area_Investment_Priority_Score created")
        
        # Store area-level features separately
        self.area_features = area_features
        
        self.merged_df = df
        return df
    
    def create_advanced_interaction_features(self):
        """Create advanced interaction features (36-45) for improved model performance."""
        print("\n" + "=" * 70)
        print("CREATING ADVANCED INTERACTION FEATURES")
        print("=" * 70)
        
        df = self.merged_df.copy()
        
        # Feature 36: Equipment_Network_Interaction
        # Interaction between equipment investment and network quality
        df['Equipment_Network_Interaction'] = (
            df['Equipment_Investment_Index'] * df['Network_Health_Index']
        ) / 100
        print("[OK] Feature 36: Equipment_Network_Interaction created")
        
        # Feature 37: Subscription_Quality_Score
        # Combine subscription utilization with voice quality
        df['Subscription_Quality_Score'] = (
            df['Subscription_Utilization_Rate'] * df['Voice_Quality_Score']
        ) / 100
        print("[OK] Feature 37: Subscription_Quality_Score created")
        
        # Feature 38: Premium_Network_Risk
        # Premium customers with poor network quality (high risk)
        df['Premium_Network_Risk'] = (
            df['Premium_Revenue_Ratio'] * (100 - df['Network_Health_Index'])
        ) / 100
        print("[OK] Feature 38: Premium_Network_Risk created")
        
        # Feature 39: CLV_Network_Alignment
        # How well customer lifetime value aligns with network quality
        df['CLV_Network_Alignment'] = (
            df['Customer_Lifetime_Value_Score'] * df['Network_Health_Index']
        ) / 100
        print("[OK] Feature 39: CLV_Network_Alignment created")
        
        # Feature 40: Revenue_Risk_Quadrant
        # Combines revenue concentration with network risk
        df['Revenue_Risk_Quadrant'] = (
            df['Revenue_Concentration_Risk'] * (100 - df['Network_Health_Index'])
        ) / 100
        print("[OK] Feature 40: Revenue_Risk_Quadrant created")
        
        # Feature 41: Engagement_Network_Synergy
        # Customer engagement combined with network performance
        df['Engagement_Network_Synergy'] = (
            df['Customer_Engagement_Score'] * df['Call_Completion_Rate']
        ) / 100
        print("[OK] Feature 41: Engagement_Network_Synergy created")
        
        # Feature 42: Growth_Network_Momentum
        # Revenue growth combined with network quality trends
        df['Growth_Network_Momentum'] = (
            df['Revenue_Growth_Momentum'] * df['Network_Health_Index']
        ) / 100
        print("[OK] Feature 42: Growth_Network_Momentum created")
        
        # Feature 43: Value_Protection_Index
        # Customer value adjusted by network risk level
        network_risk_factor = 1 - (df['Network_Health_Index'] / 100)
        df['Value_Protection_Index'] = (
            df['Customer_Lifetime_Value_Score'] * (1 - network_risk_factor)
        )
        print("[OK] Feature 43: Value_Protection_Index created")
        
        # Feature 44: Usage_Efficiency_Quality
        # Usage revenue efficiency combined with data quality
        df['Usage_Efficiency_Quality'] = (
            df['Usage_Revenue_Efficiency'] * df['Data_Quality_Score']
        ) / 100
        print("[OK] Feature 44: Usage_Efficiency_Quality created")
        
        # Feature 45: Support_Network_Load
        # Customer support load adjusted by network health
        df['Support_Network_Load'] = (
            df['Customer_Support_Load_Index'] * (100 - df['Network_Health_Index'])
        ) / 100
        print("[OK] Feature 45: Support_Network_Load created")
        
        self.merged_df = df
        return df
    
    def create_temporal_behavioral_features(self):
        """Create advanced temporal and behavioral features (46-55) for sequence analysis."""
        print("\n" + "=" * 70)
        print("CREATING ADVANCED TEMPORAL AND BEHAVIORAL FEATURES")
        print("=" * 70)
        
        df = self.merged_df.copy()
        
        # Feature 46: Usage_Volatility_Index
        # Variability in usage patterns over time
        usage_cols = ['totmou', 'mou_peav_Mean', 'mou_pead_Mean']
        available_usage = [col for col in usage_cols if col in df.columns]
        if len(available_usage) >= 2:
            usage_data = df[available_usage].fillna(0)
            df['Usage_Volatility_Index'] = usage_data.std(axis=1) / (usage_data.mean(axis=1) + 1)
        else:
            df['Usage_Volatility_Index'] = 0
        print("[OK] Feature 46: Usage_Volatility_Index created")
        
        # Feature 47: Revenue_Trajectory_Score
        # Pattern of revenue change over time (increasing, decreasing, stable)
        revenue_cols = ['rev_Mean', 'avg3rev', 'avg6rev']
        available_revenue = [col for col in revenue_cols if col in df.columns]
        if len(available_revenue) >= 3:
            rev_data = df[available_revenue].fillna(0)
            # Calculate trend: if avg6rev < avg3rev < rev_Mean = increasing
            if all(col in df.columns for col in ['avg6rev', 'avg3rev', 'rev_Mean']):
                df['Revenue_Trajectory_Score'] = (
                    (df['rev_Mean'] > df['avg3rev']).astype(int) + 
                    (df['avg3rev'] > df['avg6rev']).astype(int)
                )
            else:
                df['Revenue_Trajectory_Score'] = 0
        else:
            df['Revenue_Trajectory_Score'] = 0
        print("[OK] Feature 47: Revenue_Trajectory_Score created")
        
        # Feature 48: Network_Degradation_Rate
        # Rate of network quality decline
        network_quality = df['Network_Health_Index'].fillna(50)
        # Assume months represents time, create degradation proxy
        if 'months' in df.columns:
            tenure_normalized = df['months'].fillna(1) / df['months'].max()
            df['Network_Degradation_Rate'] = network_quality * tenure_normalized * 0.1
        else:
            df['Network_Degradation_Rate'] = network_quality * 0.05
        print("[OK] Feature 48: Network_Degradation_Rate created")
        
        # Feature 49: Customer_Aging_Factor
        # How customer behavior changes with tenure
        if 'months' in df.columns:
            tenure_years = df['months'].fillna(0) / 12
            # Interaction between tenure and usage
            avg_usage = df['totmou'].fillna(df['totmou'].mean())
            df['Customer_Aging_Factor'] = tenure_years * (avg_usage / (avg_usage.mean() + 1))
        else:
            df['Customer_Aging_Factor'] = 0
        print("[OK] Feature 49: Customer_Aging_Factor created")
        
        # Feature 50: Peak_Offpeak_Usage_Ratio
        # Usage pattern between peak and off-peak hours
        if all(col in df.columns for col in ['mou_peav_Mean', 'mou_pead_Mean']):
            peak_usage = df['mou_peav_Mean'].fillna(1)
            offpeak_usage = df['mou_pead_Mean'].fillna(1)
            df['Peak_Offpeak_Usage_Ratio'] = self.safe_divide(peak_usage, offpeak_usage, default=1)
        else:
            df['Peak_Offpeak_Usage_Ratio'] = 1
        print("[OK] Feature 50: Peak_Offpeak_Usage_Ratio created")
        
        # Feature 51: Service_Dependency_Index
        # How dependent customer is on multiple services
        service_cols = ['totmou', 'totrev', 'comp_vce_Mean', 'comp_dat_Mean']
        available_services = [col for col in service_cols if col in df.columns]
        if len(available_services) >= 2:
            service_data = df[available_services].fillna(0)
            df['Service_Dependency_Index'] = (service_data > 0).sum(axis=1) / len(available_services)
        else:
            df['Service_Dependency_Index'] = 0
        print("[OK] Feature 51: Service_Dependency_Index created")
        
        # Feature 52: Risk_Accumulation_Score
        # Accumulation of multiple risk factors
        risk_factors = 0
        if 'Network_Health_Index' in df.columns:
            risk_factors += (df['Network_Health_Index'] < 50).astype(int)
        if 'Revenue_At_Risk_Flag' in df.columns:
            risk_factors += df['Revenue_At_Risk_Flag']
        df['Risk_Accumulation_Score'] = risk_factors
        print("[OK] Feature 52: Risk_Accumulation_Score created")
        
        # Feature 53: Behavioral_Stability_Index
        # Consistency of customer behavior patterns
        behavioral_cols = ['totmou', 'totrev', 'comp_vce_Mean', 'comp_dat_Mean']
        available_behavioral = [col for col in behavioral_cols if col in df.columns]
        if len(available_behavioral) >= 2:
            behavioral_data = df[available_behavioral].fillna(0)
            # Lower std = more stable behavior
            df['Behavioral_Stability_Index'] = 1 - (behavioral_data.std(axis=1) / (behavioral_data.mean(axis=1) + 1))
            df['Behavioral_Stability_Index'] = df['Behavioral_Stability_Index'].clip(0, 1)
        else:
            df['Behavioral_Stability_Index'] = 0.5
        print("[OK] Feature 53: Behavioral_Stability_Index created")
        
        # Feature 54: Multi_Service_Adaptation
        # How well customer adapts to multiple services
        if all(col in df.columns for col in ['comp_vce_Mean', 'comp_dat_Mean']):
            voice_completion = df['complete_Mean'].fillna(0) if 'complete_Mean' in df.columns else 0
            data_completion = df['complete_Mean'].fillna(0) if 'complete_Mean' in df.columns else 0
            df['Multi_Service_Adaptation'] = (df['comp_vce_Mean'].fillna(0) + df['comp_dat_Mean'].fillna(0)) / 2
        else:
            df['Multi_Service_Adaptation'] = 0
        print("[OK] Feature 54: Multi_Service_Adaptation created")
        
        # Feature 55: Long_Term_Value_Index
        # Combined measure of long-term customer value and stability
        value_score = df['Customer_Lifetime_Value_Score'].fillna(0) if 'Customer_Lifetime_Value_Score' in df.columns else 0
        stability_score = df['Behavioral_Stability_Index'].fillna(0.5)
        df['Long_Term_Value_Index'] = value_score * stability_score
        print("[OK] Feature 55: Long_Term_Value_Index created")
        
        self.merged_df = df
        return df
    
    def select_top_15_features(self):
        """Select the top 15 priority features for the Investment Intelligence Platform."""
        print("\n" + "=" * 70)
        print("SELECTING TOP 15 INVESTMENT INTELLIGENCE FEATURES")
        print("=" * 70)
        
        top_15_features = [
            'Area_Investment_Priority_Score',
            'Infrastructure_Priority_Base',
            'Revenue_At_Risk_Flag',
            'Customer_Lifetime_Value_Score',
            'Network_Health_Index',
            'Service_Failure_Impact',
            'Revenue_Stability_Score',
            'Usage_Growth_Trend',
            'Capacity_Utilization_Rate',
            'Upsell_Potential_Score',
            'Usage_Revenue_Efficiency',
            'Area_Network_Health_Index',
            'Area_Revenue_Per_Customer',
            'Revenue_Vulnerability_Index',
            'Data_Adoption_Index'
        ]
        
        # Check which features actually exist in the dataframe
        available_features = [f for f in top_15_features if f in self.merged_df.columns]
        
        print(f"[OK] Selected {len(available_features)} out of 15 top features")
        print("Top 15 Features for Investment Intelligence Platform:")
        for i, feature in enumerate(available_features, 1):
            print(f"  {i}. {feature}")
        
        # Create a dataframe with just the top 15 features plus key identifiers
        key_columns = ['Customer_ID', 'area']
        self.top_features_df = self.merged_df[key_columns + available_features].copy()
        
        return self.top_features_df
    
    def create_validation_report(self):
        """Create a comprehensive validation report for the engineered features."""
        print("\n" + "=" * 70)
        print("CREATING VALIDATION REPORT")
        print("=" * 70)
        
        df = self.merged_df
        
        # Basic statistics for engineered features
        engineered_cols = [col for col in df.columns if any(
            keyword in col for keyword in [
                'Network_Health', 'Voice_Quality', 'Data_Quality', 'Call_Completion',
                'Revenue', 'Customer_Lifetime', 'Equipment', 'Usage_Growth',
                'Upsell', 'Infrastructure', 'Area_', 'Capacity', 'Support'
            ]
        )]
        
        print(f"\nTotal engineered features created: {len(engineered_cols)}")
        print(f"Dataset shape: {df.shape}")
        
        # Missing values analysis
        missing_values = df[engineered_cols].isnull().sum()
        if missing_values.sum() > 0:
            print("\nMissing values in engineered features:")
            print(missing_values[missing_values > 0])
        else:
            print("\n[OK] No missing values in engineered features")
        
        # Basic statistics for key features
        key_features = [
            'Network_Health_Index', 'Revenue_At_Risk_Flag', 
            'Customer_Lifetime_Value_Score', 'Area_Investment_Priority_Score'
        ]
        
        print("\nKey Features Statistics:")
        for feature in key_features:
            if feature in df.columns:
                print(f"\n{feature}:")
                print(f"  Mean: {df[feature].mean():.2f}")
                print(f"  Median: {df[feature].median():.2f}")
                print(f"  Min: {df[feature].min():.2f}")
                print(f"  Max: {df[feature].max():.2f}")
                print(f"  Std: {df[feature].std():.2f}")
        
        # Area-level summary
        if self.area_features is not None:
            print("\nArea-Level Summary:")
            print(f"Total areas: {len(self.area_features)}")
            print("\nTop 5 Areas by Investment Priority:")
            top_areas = self.area_features.nlargest(5, 'Area_Investment_Priority_Score')
            for idx, row in top_areas.iterrows():
                print(f"  {row['area']}: Priority Score = {row['Area_Investment_Priority_Score']:.2f}")
        
        return True
    
    def save_results(self):
        """Save the engineered features and area-level aggregations."""
        print("\n" + "=" * 70)
        print("SAVING FEATURE ENGINEERING RESULTS")
        print("=" * 70)
        
        # Save complete engineered dataset
        output_path = self.output_dir / 'engineered_telecom_features.csv'
        self.merged_df.to_csv(output_path, index=False)
        print(f"[OK] Complete engineered dataset saved to: {output_path}")
        
        # Save top 15 features
        if hasattr(self, 'top_features_df'):
            top_features_path = self.output_dir / 'top_15_investment_features.csv'
            self.top_features_df.to_csv(top_features_path, index=False)
            print(f"[OK] Top 15 features saved to: {top_features_path}")
        
        # Save area-level features
        if self.area_features is not None:
            area_features_path = self.output_dir / 'area_level_investment_features.csv'
            self.area_features.to_csv(area_features_path, index=False)
            print(f"[OK] Area-level features saved to: {area_features_path}")
        
        # Save feature documentation
        self.save_feature_documentation()
        
        return True
    
    def save_feature_documentation(self):
        """Save documentation for all engineered features."""
        doc_path = self.output_dir / 'feature_documentation.md'
        
        documentation = """# Telecom Investment Intelligence Platform - Feature Documentation

## Overview
This document describes the 35 business-focused engineered features created for the Telecom Investment Intelligence Platform.

## Feature Categories

### Network Quality Features (1-8)
These features assess network performance and quality for infrastructure investment decisions.

#### 1. Network_Health_Index
- **Formula**: `100 - ((drop_vce_Mean + drop_dat_Mean) * 10 + (blck_vce_Mean + blck_dat_Mean) * 5)`
- **Purpose**: Overall network performance score (0-100)
- **Business Impact**: Direct ROI metric for network investments

#### 2. Voice_Quality_Score
- **Formula**: `100 - (drop_vce_Mean * 15 + blck_vce_Mean * 8)`
- **Purpose**: Voice-specific network quality
- **Business Impact**: Voice quality directly impacts customer satisfaction

#### 3. Data_Quality_Score
- **Formula**: `100 - (drop_dat_Mean * 15 + blck_dat_Mean * 8)`
- **Purpose**: Data-specific network quality
- **Business Impact**: Data services have higher margins

#### 4. Call_Completion_Rate
- **Formula**: `complete_Mean / attempt_Mean * 100`
- **Purpose**: Percentage of successfully completed calls
- **Business Impact**: Operational efficiency metric

#### 5. Network_Risk_Level
- **Formula**: Categorical (CRITICAL/WARNING/HEALTHY based on Network_Health_Index)
- **Purpose**: Categorical risk assessment
- **Business Impact**: Simplifies investment priorities

#### 6. Peak_Hour_Quality
- **Formula**: `(peak_vce_Mean + peak_dat_Mean) / (mou_peav_Mean + mou_pead_Mean)`
- **Purpose**: Network quality during high-demand periods
- **Business Impact**: Identifies capacity constraints

#### 7. Service_Failure_Impact
- **Formula**: `(drop_vce_Mean + drop_dat_Mean) * totrev`
- **Purpose**: Revenue at risk due to network failures
- **Business Impact**: Quantifies financial impact

#### 8. Customer_Support_Load_Index
- **Formula**: `custcare_Mean / (totmou / 1000)`
- **Purpose**: Support calls per 1000 minutes of usage
- **Business Impact**: High support costs indicate network issues

### Revenue Risk Features (9-14)
These features identify revenue protection opportunities and risks.

#### 9. Revenue_Stability_Score
- **Formula**: `1 - STDDEV(avg3rev, avg6rev, rev_Mean) / MEAN(avg3rev, avg6rev, rev_Mean)`
- **Purpose**: Consistency of revenue over time (0-1)
- **Business Impact**: Identifies volatile revenue patterns

#### 10. Revenue_At_Risk_Flag
- **Formula**: `IF(change_rev < -20 AND Network_Health_Index < 75, 1, 0)`
- **Purpose**: Customers with declining revenue AND poor network quality
- **Business Impact**: Priority intervention targets

#### 11. Usage_Revenue_Efficiency
- **Formula**: `totrev / totmou`
- **Purpose**: Revenue generated per minute of usage
- **Business Impact**: Pricing strategy optimization

#### 12. Revenue_Vulnerability_Index
- **Formula**: `(rev_Mean - avg3rev) / rev_Mean * Network_Risk_Factor`
- **Purpose**: Combines revenue trends with network quality risks
- **Business Impact**: Early warning system

#### 13. Premium_Revenue_Ratio
- **Formula**: `ovrrev_Mean / rev_Mean`
- **Purpose**: Percentage of revenue from overage
- **Business Impact**: Identifies premium service customers

#### 14. Revenue_Concentration_Risk
- **Formula**: `rev_Mean / totrev`
- **Purpose**: Individual customer revenue contribution
- **Business Impact**: Identifies over-reliance risks

### Customer Value Features (15-20)
These features assess customer value for segmentation and retention.

#### 15. Customer_Lifetime_Value_Score
- **Formula**: `rev_Mean * (months / 12) * (1 + (hnd_price / 100))`
- **Purpose**: Estimated lifetime value
- **Business Impact**: Prioritizes retention investments

#### 16. Equipment_Investment_Index
- **Formula**: `hnd_price * (1 - (eqpdays / 3650))`
- **Purpose**: Equipment value adjusted for age
- **Business Impact**: Newer equipment indicates lower churn risk

#### 17. Subscription_Utilization_Rate
- **Formula**: `actvsubs / uniqsubs`
- **Purpose**: Active subscriptions as percentage of total
- **Business Impact**: Identifies underutilized potential

#### 18. Revenue_Per_Subscription
- **Formula**: `totrev / uniqsubs`
- **Purpose**: Average revenue per subscription
- **Business Impact**: Pricing and plan optimization

#### 19. Customer_Engagement_Score
- **Formula**: `(totmou / months) * (complete_Mean / attempt_Mean) * (1 + (phones / 10))`
- **Purpose**: Engagement combining usage, completion, and devices
- **Business Impact**: Highly engaged customers are more loyal

#### 20. Credit_Quality_Adjusted_Value
- **Formula**: `CLV_Score * IF(creditcd == 'Y', 1.2, 0.8)`
- **Purpose**: Customer value adjusted for credit quality
- **Business Impact**: Credit quality affects payment reliability

### Growth Potential Features (21-26)
These features identify growth opportunities for upsell and revenue growth.

#### 21. Usage_Growth_Trend
- **Formula**: `(avg6mou - avg3mou) / avg3mou`
- **Purpose**: Percentage change in usage
- **Business Impact**: Identifies growing customers

#### 22. Revenue_Growth_Momentum
- **Formula**: `(avg6rev - avg3rev) / avg3rev`
- **Purpose**: Revenue growth acceleration
- **Business Impact**: Highlights effective campaigns

#### 23. Data_Adoption_Index
- **Formula**: `da_Mean / (da_Mean + mou_Mean)`
- **Purpose**: Data usage as percentage of total
- **Business Impact**: Data adoption indicates upsell potential

#### 24. Upsell_Potential_Score
- **Formula**: `(Usage_Growth_Trend * 0.4) + (Data_Adoption_Index * 0.3) + (hnd_price / 200 * 0.3)`
- **Purpose**: Composite score for premium service readiness
- **Business Impact**: Direct targeting for sales teams

#### 25. Cross_Sell_Opportunity
- **Formula**: `(uniqsubs - actvsubs) * rev_Mean`
- **Purpose**: Revenue potential from activating inactive subscriptions
- **Business Impact**: Quick wins through activation campaigns

#### 26. Tenure_Based_Growth_Potential
- **Formula**: `Tenure_Multiplier * Usage_Growth_Trend`
- **Purpose**: Growth potential adjusted for lifecycle stage
- **Business Impact**: New customers have higher growth potential

### Infrastructure Investment Features (27-31)
These features support capital allocation decisions.

#### 27. Infrastructure_Priority_Base
- **Formula**: `Network_Health_Index * rev_Mean / 1000`
- **Purpose**: Base score for infrastructure priority
- **Business Impact**: Aggregated at area level for ROI calculation

#### 28. Capacity_Utilization_Rate
- **Formula**: `(peak_vce_Mean + peak_dat_Mean) / (totmou / 30)`
- **Purpose**: Peak usage as percentage of average daily usage
- **Business Impact**: Identifies areas requiring expansion

#### 29. Revenue_At_Risk_Amount
- **Formula**: `Revenue_At_Risk_Flag * rev_Mean`
- **Purpose**: Financial justification for tower investments
- **Business Impact**: ROI calculation for infrastructure projects

#### 30. Support_Cost_Driver_Index
- **Formula**: `custcare_Mean * (1 - Network_Health_Index/100)`
- **Purpose**: Support costs driven by network quality
- **Business Impact**: Quantifies operational cost savings

#### 31. Roaming_Revenue_Opportunity
- **Formula**: `roam_Mean * rev_Mean / totmou`
- **Purpose**: Revenue potential from roaming services
- **Business Impact**: Identifies underdeveloped roaming revenue

### Area-Level Aggregation Features (32-35)
These features support regional investment decisions.

#### 32. Area_Investment_Priority_Score
- **Formula**: Weighted composite of area-level metrics
- **Purpose**: Executive-level capital allocation metric
- **Business Impact**: Direct regional investment prioritization

#### 33. Area_Network_Health_Index
- **Formula**: `AVG(Network_Health_Index) GROUP BY area`
- **Purpose**: Average network quality by service area
- **Business Impact**: Regional performance comparison

#### 34. Area_Revenue_Per_Customer
- **Formula**: `SUM(totrev) / COUNT(Customer_ID) GROUP BY area`
- **Purpose**: Average revenue per customer by area
- **Business Impact**: Identifies high-value regions

#### 35. Area_Customer_Density
- **Formula**: `COUNT(Customer_ID) GROUP BY area`
- **Purpose**: Customer concentration by area
- **Business Impact**: Network planning optimization

## Data Quality Notes
- All features handle missing values through safe division and fillna operations
- Features are clipped to appropriate ranges (e.g., 0-100 for scores)
- Division by zero is handled with default values
- Categorical features use safe mapping

## Top 15 Investment Intelligence Features
1. Area_Investment_Priority_Score
2. Infrastructure_Priority_Base
3. Revenue_At_Risk_Flag
4. Customer_Lifetime_Value_Score
5. Network_Health_Index
6. Service_Failure_Impact
7. Revenue_Stability_Score
8. Usage_Growth_Trend
9. Capacity_Utilization_Rate
10. Upsell_Potential_Score
11. Usage_Revenue_Efficiency
12. Area_Network_Health_Index
13. Area_Revenue_Per_Customer
14. Revenue_Vulnerability_Index
15. Data_Adoption_Index

## Usage Recommendations
- Use Area_Investment_Priority_Score for regional capital allocation
- Use Revenue_At_Risk_Flag for immediate revenue protection interventions
- Use Customer_Lifetime_Value_Score for retention program targeting
- Use Network_Health_Index for overall network performance tracking
- Use Upsell_Potential_Score for sales team targeting
"""
        
        with open(doc_path, 'w') as f:
            f.write(documentation)
        
        print(f"[OK] Feature documentation saved to: {doc_path}")
        
        return True
    
    def run_complete_pipeline(self):
        """Execute the complete feature engineering pipeline."""
        print("=" * 70)
        print("TELECOM INVESTMENT INTELLIGENCE PLATFORM")
        print("BUSINESS FEATURE ENGINEERING PIPELINE")
        print("=" * 70)
        
        # Step 1: Load data
        if not self.load_data():
            return False
        
        # Step 2: Create feature categories
        self.create_network_quality_features()
        self.create_revenue_risk_features()
        self.create_customer_value_features()
        self.create_growth_potential_features()
        self.create_infrastructure_investment_features()
        self.create_advanced_interaction_features()
        self.create_temporal_behavioral_features()
        self.create_area_level_features()
        
        # Step 3: Select top features
        self.select_top_15_features()
        
        # Step 4: Validation
        self.create_validation_report()
        
        # Step 5: Save results
        self.save_results()
        
        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"Total engineered features: 35")
        print(f"Top investment features: 15")
        print(f"Areas analyzed: {len(self.area_features) if self.area_features is not None else 'N/A'}")
        
        return True

def main():
    """Main execution function."""
    engineer = BusinessFeatureEngineer()
    success = engineer.run_complete_pipeline()
    
    if success:
        print("\n[SUCCESS] Business feature engineering completed successfully!")
    else:
        print("\n[ERROR] Business feature engineering failed. Please check the error messages above.")

if __name__ == "__main__":
    main()