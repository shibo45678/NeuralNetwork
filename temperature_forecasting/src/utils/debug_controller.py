import os
from datetime import datetime
import pickle
import inspect
import hashlib
import warnings
import glob


class DebugController:
    """专门用于调试的检查点系统，- 修复序列化"""

    def __init__(self, debug_dir='debug_sessions'):
        self.debug_dir = debug_dir
        os.makedirs(debug_dir, exist_ok=True)
        self.stage_order: list[str] = ['after_data_preparation',
                                       'after_model_build_cnn', 'after_training_cnn',
                                       'after_model_lstm1', 'after_model_lstm2',
                                       'after_models_compare']

    def get_session_info(self):
        """获取会话信息"""
        info = []
        for stage in self.stage_order:
            stage_file = os.path.join(self.debug_dir, f'session_{stage}.pkl')
            stage_hash_file = os.path.join(self.debug_dir, f'hash_{stage}.txt')

            if os.path.exists(stage_file) and os.path.exists(stage_hash_file):
                status = "有效"
            else:
                status = "缺失"
            info.append(f"{stage}:{status}")

        return '\n'.join(info) if info else "无保存的会话"

    def continue_from_breakpoint(self, breakpoint_name, current_locals, target_file=None):
        """""""""依赖链式检查，从断点继续执行，成功返回阶段名称，失败返回None"""

        try:
            # 无效
            if not self._is_stage_valid(breakpoint_name, target_file):
                self._clear_stages_from(breakpoint_name)
                print(f"当前阶段 {breakpoint_name} 文件缺失或更改，已清除当前及后续阶段")
                print(f"查找可用阶段...")

                return self._find_available_stage(breakpoint_name, current_locals, target_file)

            # 有效
            else:
                print(f"当前阶段 {breakpoint_name} 有效，加载当前 {breakpoint_name} 阶段...")
                if self._load_specific_stage(breakpoint_name, current_locals, target_file):
                    return breakpoint_name  # 返回成功加载的阶段名称
                else:
                    return None

        except Exception as e:
            print(f"继续执行断点 {breakpoint_name} 失败：{str(e)}")
        return None


    def _is_stage_valid(self, stage_name=None, target_file=None):
        stage_file = os.path.join(self.debug_dir, f'session_{stage_name}.pkl')
        stage_hash_file = os.path.join(self.debug_dir, f'hash_{stage_name}.txt')

        # 检查文件是否存在
        if not os.path.exists(stage_file):
            print(f"阶段 {stage_name} 会话文件缺失")
            return False
        if not os.path.exists(stage_hash_file):
            print(f"阶段 {stage_name} 哈希文件缺失")
            return False

        # 检查代码是否修改
        current_hash = self.get_code_hash(stage_name, target_file)
        try:
            with open(stage_hash_file, 'r') as f:
                saved_hash = f.read().strip()

            if current_hash != saved_hash:
                print(f"阶段 {stage_name}:代码已修改（{saved_hash[:8]} ->{current_hash[:8]}）")
                return False
            else:
                print(f"阶段 {stage_name}：代码未修改，哈希值一致")
                return True

        except Exception as e:
            print(f"阶段 {stage_name}: 哈希文件读取失败: {e}")
            return False

    def get_code_hash(self, stage_name=None, target_file=None):
        """""""计算当前代码的哈希值，检测是否修改"""
        if target_file is None:
            frame = inspect.currentframe()
            try:
                caller_frame = frame.f_back.f_back  # 跳过自身和调用函数
                target_file = inspect.getfile(caller_frame)
            finally:
                del frame  # 避免循环引用
        try:
            with open(target_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 指定阶段名称，提取该阶段相关代码
            if stage_name:
                content = self._extract_stage_code_by_markers(lines, stage_name)
            else:
                content = ''.join(lines)  # 没有阶段，就整个main.py 文件

            return hashlib.md5(content.encode()).hexdigest()

        except FileNotFoundError:
            warnings.warn(f"无法找到文件{target_file}")
            return "unknown"

    def _extract_stage_code_by_markers(self, lines, stage_name):
        """通过注释标记提取特定阶段的代码"""
        start_marker = f"# STAGE_START:{stage_name}"  # 在main.py内部添加注释
        end_marker = f"# STAGE_END:{stage_name}"

        print(f"开始标记: {start_marker}")
        print(f"结束标记: {end_marker}")

        stage_lines = []
        in_stage = False
        found_marker = False

        for line in lines:
            if start_marker in line:
                in_stage = True
                found_marker = True
                continue  # 不包含标记行本身
            elif end_marker in line:
                break  # 遇到结束标记就停止
            elif in_stage:
                stage_lines.append(line)  # 在收集模式下in_stage=True：记录数据，保存代码

        content = ''.join(stage_lines)
        print(f"提取的代码长度: {len(content)} 字符")

        # 如果找到了标记，返回阶段代码；否则返回整个文件
        return content if found_marker else ''.join(lines)  # 原表有换行会保留

    def _load_specific_stage(self, stage_name, current_locals, target_file):
        """加载指定有效阶段的数据"""
        try:
            stage_file = os.path.join(self.debug_dir, f'session_{stage_name}.pkl')

            # 文件存在且代码未修改，加载当前阶段数据
            with open(stage_file, 'rb') as f:
                session_data = pickle.load(f)
            restored_locals = self.__restore_locals(session_data['locals'], current_locals)
            current_locals.update(restored_locals)
            print(f"阶段 {stage_name} 已完成加载")
            return True

        except Exception as e:
            print(f"加载阶段 {stage_name} 失败：{str(e)}")
            return False

    def _find_available_stage(self, breakpoint_name, current_locals, target_file):
        """只负责无效(删除/哈希吗改动）数据的查找可用"""
        try:
            current_index = self.stage_order.index(breakpoint_name)

            # 从当前阶段的前一个开始往前找
            for i in range(current_index - 1, -1, -1):
                prev_stage = self.stage_order[i]

                # 检查备用阶段是否有效（文件存在且代码未修改）
                if self._is_stage_valid(prev_stage, target_file):
                    print(f"找到备用阶段: {prev_stage},并加载...")
                    if self._load_specific_stage(prev_stage, current_locals, target_file):
                        print(f"已成功加载备用阶段: {prev_stage}")
                        return prev_stage

            # 没有找到任何可用阶段
            self.clear_sessions()
            print("没有找到任何备用阶段，删除所有阶段文件，重新执行程序")
            return None

        except ValueError:
            print(f"未知阶段: {breakpoint_name}")
            return None

    def clear_sessions(self):
        """清除所有阶段文件会话"""
        file_patterns = ['session_*.pkl', 'hash_*.txt']

        files_removed = 0
        for pattern in file_patterns:
            for file_path in glob.glob(os.path.join(self.debug_dir, pattern)):
                try:
                    os.remove(file_path)
                    files_removed += 1
                    print(f"已删除: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"删除文件失败 {file_path}: {e}")

        print(f"已清除{files_removed}个调试会话文件")
        return files_removed > 0

    def _clear_stages_from(self, start_stage):
        """从指定阶段开始清除所有后续阶段的文件（只清除文件，不涉及加载逻辑）"""
        try:
            start_index = self.stage_order.index(start_stage)
            stages_to_clear = self.stage_order[start_index:]
        except ValueError:
            print(f"未知起始阶段: {start_stage}")
            stages_to_clear = self.stage_order

        files_removed = 0
        for stage in stages_to_clear:
            stage_file = os.path.join(self.debug_dir, f'session_{stage}.pkl')
            stage_hash_file = os.path.join(self.debug_dir, f'hash_{stage}.txt')

            for file_path in [stage_file, stage_hash_file]:  # 创建了一个包含这两个字符串stage_file，stage_hash_file的列表
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        files_removed += 1
                        print(f"已删除: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"删除文件失败 {file_path}: {str(e)}")

        print(f"从阶段 {start_stage} 开始清除，共移除{files_removed}个文件")

    def _sanitize_locals(self, locals_dict):
        """清理locals，移除不可序列化的对象"""
        sanitized = {}
        skipped_vars = []

        for key, value in locals_dict.items():
            if key.startswith('_'):
                continue
            try:
                pickle.dumps(value)
                sanitized[key] = value
            except (pickle.PickleError, TypeError, AttributeError) as e:
                skipped_vars.append((key, type(value).__name__))
                # 对于不可序列化的对象，保存其类型信息
                sanitized[key] = {
                    '__debug_placeholder__': True,
                    'type': type(value).__name__,
                    'repr': repr(value)[:200]
                }

        if skipped_vars:
            print(f"跳过不可序列化的变量: {skipped_vars}")

        return sanitized

    def __restore_locals(self, saved_locals, current_locals):
        restored = {}
        for key, value in saved_locals.items():
            if isinstance(value, dict) and value.get('__debug_placeholder__'):
                print(f"变量 {key} 无法恢复，需要重新创建")
                restored[key] = None
            else:
                restored[key] = value

        return restored

    def save_debug_session(self, locals_dict, breakpoint_name, target_file=None):
        """""""""分阶段存储会话的所有变量，保存代码哈希"""
        # 为每个阶段创建独立文件
        stage_file = os.path.join(self.debug_dir, f'session_{breakpoint_name}.pkl')
        stage_hash_file = os.path.join(self.debug_dir, f'hash_{breakpoint_name}.txt')

        # 保存这个阶段的代码哈希
        current_hash = self.get_code_hash(breakpoint_name, target_file)
        try:
            with open(stage_hash_file, 'w') as f:
                f.write(current_hash)
            print(f"阶段 {breakpoint_name} 哈希码已保存: {current_hash[:8]}...")

        except Exception as e:
            print(f"阶段 {breakpoint_name} 哈希码保存失败：{str(e)}")


        # 清理locals
        sanitized_locals = self._sanitize_locals(locals_dict)

        # 保存会话
        session_data = {
            'breakpoint': breakpoint_name,
            'locals': sanitized_locals,
            'timestamp': datetime.now().isoformat(),
            # 'code_hash': current_hash
        }
        try:
            with open(stage_file, 'wb') as f:
                pickle.dump(session_data, f)
            print(f"阶段 {breakpoint_name} 会话已保存")
            return True

        except Exception as e:
            print(f"阶段 {breakpoint_name} 会话保存失败：{e}")
            return False

#
# def get_dependencies_hash(self, stage_name=None, target_file=None):
#     """计算主文件及其依赖文件的哈希"""
#     if target_file is None:
#         frame = inspect.currentframe()
#         try:
#             caller_frame = frame.f_back.f_back
#             target_file = inspect.getfile(caller_frame)
#         finally:
#             del frame
#
#     # 定义需要检查的关键文件
#     project_root = os.path.dirname(target_file)
#     key_files = [
#         target_file,  # 主文件
#         os.path.join(project_root, 'src', 'load_data.py'),
#         os.path.join(project_root, 'src', 'model_builder.py'),
#         os.path.join(project_root, 'src', 'utils.py'),
#         # 添加其他重要依赖文件
#     ]
#
#     hash_list = []
#     for file_path in key_files:
#         if os.path.exists(file_path):
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     content = f.read()
#                 file_hash = hashlib.md5(content.encode()).hexdigest()
#                 hash_list.append(f"{os.path.basename(file_path)}:{file_hash}")
#             except Exception as e:
#                 print(f"读取文件 {file_path} 失败: {e}")
#
#     combined_hash = hashlib.md5(''.join(hash_list).encode()).hexdigest()
#     print(f"依赖哈希: 检查了 {len(hash_list)} 个关键文件")
#     return combined_hash