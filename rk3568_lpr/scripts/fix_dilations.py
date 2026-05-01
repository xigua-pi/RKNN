import onnx

def fix_maxpool_dilations(input_path, output_path):
    model = onnx.load(input_path)
    changed = 0
    for node in model.graph.node:
        if node.op_type == "MaxPool":
            # 找出所有名为 "dilations" 的属性索引
            to_remove = [i for i, attr in enumerate(node.attribute) if attr.name == "dilations"]
            if to_remove:
                changed += len(to_remove)
                # 按照索引逆序删除 dilations 属性，避免位置错乱
                for idx in reversed(to_remove):
                    del node.attribute[idx]
    if changed:
        onnx.save(model, output_path)
        print(f"Removed {changed} dilations attributes; saved to {output_path}")
    else:
        print("No MaxPool dilations found; model unchanged.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python fix_dilations.py <input.onnx> <output.onnx>")
    else:
        fix_maxpool_dilations(sys.argv[1], sys.argv[2])