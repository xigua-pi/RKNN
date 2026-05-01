import onnx

def fix_maxpool_attributes(input_path: str, output_path: str):
    model = onnx.load(input_path)
    changed = 0
    for node in model.graph.node:
        if node.op_type == "MaxPool":
            # 删除 dilations 属性
            indices = [i for i, attr in enumerate(node.attribute) if attr.name == "dilations"]
            for idx in reversed(indices):
                del node.attribute[idx]
                changed += 1
            # 修复 strides 属性长度为 2
            for attr in node.attribute:
                if attr.name == "strides":
                    vals = list(attr.ints)
                    if len(vals) != 2:
                        if len(vals) == 0:
                            new_vals = [1, 1]       # 默认步长
                        elif len(vals) == 1:
                            new_vals = [vals[0], vals[0]]
                        else:
                            new_vals = [vals[0], vals[1]]
                        attr.ClearField("ints")
                        attr.ints.extend(new_vals)
                        changed += 1
    onnx.save(model, output_path)
    print(f"Fixed {changed} attributes; saved to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python fix_maxpool_attrs.py <input.onnx> <output.onnx>")
    else:
        fix_maxpool_attributes(sys.argv[1], sys.argv[2])