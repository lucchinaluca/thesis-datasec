"""
Pipeline 2: Clustering and Anomaly Detection
This pipeline uses modern unsupervised learning approaches including PyOD's deep learning models
and ensemble methods to detect anomalies and classify attack patterns.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.lscp import LSCP
from pyod.models.feature_bagging import FeatureBagging

class ModernAnomalyDetector:
    """Modern anomaly detection using an ensemble of PyOD detectors"""
    def __init__(self, contamination=0.1, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        
        # Initialize modern detectors
        self.deep_svdd = DeepSVDD(contamination=contamination, random_state=random_state)
        self.so_gaal = SO_GAAL(contamination=contamination, random_state=random_state)
        
        # Feature bagging ensemble
        self.feature_bagging = FeatureBagging(
            base_estimator=DeepSVDD(contamination=contamination),
            n_estimators=5,
            contamination=contamination,
            random_state=random_state
        )
        
        # LSCP ensemble (Locally Selective Combination)
        self.detector_list = [DeepSVDD(), SO_GAAL()]
        self.lscp = LSCP(
            detector_list=self.detector_list,
            contamination=contamination,
            random_state=random_state
        )
        
    def fit_predict(self, X):
        """Fit detectors and return ensemble predictions"""
        # Get predictions from each detector
        deep_svdd_labels = self.deep_svdd.fit_predict(X)
        so_gaal_labels = self.so_gaal.fit_predict(X)
        feature_bagging_labels = self.feature_bagging.fit_predict(X)
        lscp_labels = self.lscp.fit_predict(X)
        
        # Combine predictions using majority voting
        predictions = np.vstack([
            deep_svdd_labels,
            so_gaal_labels,
            feature_bagging_labels,
            lscp_labels
        ])
        
        # Return majority vote (-1 for anomaly, 1 for normal)
        return np.sign(predictions.mean(axis=0))
    
    def decision_scores(self, X):
        """Get anomaly scores from all detectors"""
        return {
            'deep_svdd': self.deep_svdd.decision_scores_,
            'so_gaal': self.so_gaal.decision_scores_,
            'feature_bagging': self.feature_bagging.decision_scores_,
            'lscp': self.lscp.decision_scores_
        }

class Pipeline2:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clusterer = HDBSCAN(
            min_cluster_size=5,
            min_samples=3,
            cluster_selection_epsilon=0.1
        )
        self.anomaly_detector = ModernAnomalyDetector(contamination=0.1)

    def prepare_features(self, df):
        """Extract and prepare features for anomaly detection"""
        features = {}
        
        # Process temporal features
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df_sorted = df.sort_values('timestamp')
                time_diffs = df_sorted.groupby('src_ip')['timestamp'].diff().dt.total_seconds()
                
                features.update({
                    'hour_sin': np.sin(2 * np.pi * df['timestamp'].dt.hour/24),
                    'hour_cos': np.cos(2 * np.pi * df['timestamp'].dt.hour/24),
                    'day_sin': np.sin(2 * np.pi * df['timestamp'].dt.dayofweek/7),
                    'day_cos': np.cos(2 * np.pi * df['timestamp'].dt.dayofweek/7),
                    'is_weekend': df['timestamp'].dt.dayofweek.isin([5,6]).astype(int)
                })
                
                # Time difference features
                if not time_diffs.isna().all():
                    time_stats = df.groupby('src_ip')['timestamp'].agg([
                        ('time_mean_diff', lambda x: x.diff().dt.total_seconds().mean()),
                        ('time_std_diff', lambda x: x.diff().dt.total_seconds().std()),
                        ('time_max_diff', lambda x: x.diff().dt.total_seconds().max())
                    ]).fillna(0)
                    features.update(time_stats.to_dict())
                    
            except Exception as e:
                print(f"Warning: Could not process temporal features: {e}")

        # Network behavior features
        if all(col in df.columns for col in ['src_ip', 'dst_ip', 'dst_port']):
            try:
                conn_stats = df.groupby('src_ip').agg({
                    'dst_ip': ['nunique', 'count'],
                    'dst_port': ['nunique', 'mean', 'std']
                }).fillna(0)
                
                features.update({
                    'unique_targets': conn_stats[('dst_ip', 'nunique')],
                    'total_connections': conn_stats[('dst_ip', 'count')],
                    'unique_ports': conn_stats[('dst_port', 'nunique')],
                    'mean_port': conn_stats[('dst_port', 'mean')],
                    'port_std': conn_stats[('dst_port', 'std')]
                })
                
                # Calculate port entropy
                port_dist = df.groupby(['src_ip', 'dst_port']).size().unstack(fill_value=0)
                port_probs = port_dist.div(port_dist.sum(axis=1), axis=0)
                features['port_entropy'] = -(port_probs * np.log2(port_probs + 1e-10)).sum(axis=1)
                
            except Exception as e:
                print(f"Warning: Could not calculate network features: {e}")
        
        # Protocol and categorical features
        categorical_cols = [col for col in df.columns if col.startswith('protocol_')]
        for col in categorical_cols:
            try:
                dummies = pd.get_dummies(df[col], prefix=col)
                features.update(dummies)
            except Exception as e:
                print(f"Warning: Could not process categorical feature {col}: {e}")
        
        # Combine all features
        feature_df = pd.DataFrame(features)
        feature_df = feature_df.fillna(0)  # Handle any missing values
        
        return feature_df

    def detect_anomalies(self, features):
        """Detect anomalies using modern PyOD methods"""
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Get predictions and scores
        predictions = self.anomaly_detector.fit_predict(scaled_features)
        scores = self.anomaly_detector.decision_scores(scaled_features)
        
        return {
            'predictions': predictions,
            'scores': scores
        }

    def analyze_clusters(self, features):
        """Analyze clusters for patterns"""
        scaled_features = self.scaler.fit_transform(features)
        clusters = self.clusterer.fit_predict(scaled_features)
        
        cluster_stats = {}
        for cluster in np.unique(clusters):
            if cluster == -1:  # Noise points
                continue
                
            cluster_features = features[clusters == cluster]
            stats = {
                'size': len(cluster_features),
                'temporal_regularity': self._calculate_temporal_regularity(cluster_features),
                'port_diversity': self._calculate_port_diversity(cluster_features),
                'connection_density': self._calculate_connection_density(cluster_features)
            }
            cluster_stats[cluster] = stats
            
        return clusters, cluster_stats
    
    def _calculate_temporal_regularity(self, cluster_features):
        """Calculate how regular the temporal patterns are"""
        if 'time_std_diff' in cluster_features.columns:
            return 1 / (1 + cluster_features['time_std_diff'].mean())
        return None
    
    def _calculate_port_diversity(self, cluster_features):
        """Calculate diversity of ports used"""
        if 'port_entropy' in cluster_features.columns:
            return cluster_features['port_entropy'].mean()
        return None
    
    def _calculate_connection_density(self, cluster_features):
        """Calculate density of connections"""
        if 'total_connections' in cluster_features.columns and 'unique_targets' in cluster_features.columns:
            return cluster_features['total_connections'].mean() / (1 + cluster_features['unique_targets'].mean())
        return None

if __name__ == "__main__":
    # Load data
    df = pd.read_csv('omnipot.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Initialize pipeline
    pipeline = Pipeline2()
    
    # Extract features
    features = pipeline.prepare_features(df)
    
    # Detect anomalies
    anomaly_results = pipeline.detect_anomalies(features)
    
    # Analyze clusters
    clusters, cluster_stats = pipeline.analyze_clusters(features)
    
    # Print results
    print("\nAnomaly Detection Results:")
    n_anomalies = (anomaly_results['predictions'] == -1).sum()
    print(f"Total anomalies detected: {n_anomalies}")
    
    print("\nCluster Analysis:")
    for cluster_id, stats in cluster_stats.items():
        print(f"\nCluster {cluster_id}:")
        print(f"Size: {stats['size']}")
        if stats['temporal_regularity']:
            print(f"Temporal regularity: {stats['temporal_regularity']:.3f}")
        if stats['port_diversity']:
            print(f"Port diversity: {stats['port_diversity']:.3f}")
        if stats['connection_density']:
            print(f"Connection density: {stats['connection_density']:.3f}")
