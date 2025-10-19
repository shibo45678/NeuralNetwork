


    @validate_input(validate_y=False)
    def transform(self, X):
        X_=pd.DataFrame(X) if not isinstance(X,pd.DataFrame) else X.copy()

        for col,config in self.scaling_config.items():
            if col not in X_.columns:
                continue

            if method.startswith('constant'):
                # 常数列特殊处理
                constant_value = config.get('constant_value', 0)

                if method == 'constant_minmax':
                    # MinMax规则：设为0.5
                    X_transformed[col] = 0.5
                elif method in ['constant_standard', 'constant_robust']:
                    # Standard/Robust规则：设为0
                    X_transformed[col] = 0

                print(f"常数列 '{col}': 原始值={constant_value}, 转换值={X_transformed[col].iloc[0]}")


            elif config['method'] == 'skip':
                continue
            else:
                scaler = self.fitted_scalers.get(col)
                if scaler is not None:
                    non_null_mask = X_[col].notna()
                    if non_null_mask.any():
                        transformed_vals = scaler.transform(X_.loc[non_null_mask,col].values.reshape(-1,1)).flatten()
                        X_.loc[non_null_mask,col] = transformed_vals
        return X_


    def get_scaling_report(self):
        report = {
            'algorithm': self.algorithm,
            'scaling_config': self.scaling_config,
            'summary': {
                'total_features': len(self.scaling_config),
                'standard_count': len([c for c in self.scaling_config.values() if c['method'] == 'standard']),
                'minmax_count': len([c for c in self.scaling_config.values() if c['method'] == 'minmax']),
                'robust_count': len([c for c in self.scaling_config.values() if c['method'] == 'robust']),
                'skip_count': len([c for c in self.scaling_config.values() if c['method'] == 'skip'])
            }
        }
        return report
#
#     def inverse_transform(self, X):
#         """逆转换 - 包括常数列的恢复"""
#         X_original = X.copy()
#
#         for col, config in self.scaling_config.items():
#             if col not in X_original.columns:
#                 continue
#
#             method = config['method']
#
#             if method.startswith('constant'):
#                 # 恢复常数列的原始值
#                 constant_value = config.get('constant_value', 0)
#                 X_original[col] = constant_value
#             else:
#                 # 正常逆转换
#                 scaler = self.fitted_scalers.get(col)
#                 if scaler is not None:
#                     non_null_mask = X_original[col].notna()
#                     if non_null_mask.any():
#                         original_vals = scaler.inverse_transform(
#                             X_original.loc[non_null_mask, col].values.reshape(-1, 1)
#                         ).flatten()
#                         X_original.loc[non_null_mask, col] = original_vals
#
#         return X_original
#
