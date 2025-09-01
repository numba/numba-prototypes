from copy import deepcopy
import shutil
import subprocess
import tempfile
import os
from mlir.passmanager import PassManager
from mlir import ir

class MLIRVerifier:
    def __init__(self, mlir_module, mlir_opt_path="mlir-opt", mlir_translate_path="mlir-translate"):
        self.mlir_module = mlir_module
        self.mlir_opt_path = mlir_opt_path
        self.mlir_translate_path = mlir_translate_path

    def verify_passes(self, passes, output_dir=None, output_info=True):
        """
        Apply mlir-opt passes in sequence, writing each output if output_dir is given.
        Returns the final MLIR text if all passes succeed, else None.
        """
        mlir_text = str(self.mlir_module)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        os.rmdir(file_path)
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
            with open(os.path.join(output_dir, f"input_module.mlir"), "w") as f:
                f.write(mlir_text)
                print(f"Input module written to: {os.path.join(output_dir, f'input_module.mlir')}")

        for idx, p in enumerate(passes):
            cmd = [self.mlir_opt_path, f"--{p}"]
            if output_info:
                print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, input=mlir_text, capture_output=True, text=True)
            if result.returncode != 0 and output_info:
                print(f"Error running pass '{p}':")
                print(result.stderr)
                return None
            else:
                if output_info:
                    print(f"Pass '{p}' succeeded.")
                mlir_text = result.stdout
                if mlir_text.strip() == "":
                    mlir_text = result.stderr

                if output_dir is not None:
                    pass_filename = os.path.join(output_dir, f"pass_{idx+1}_{p}.mlir")
                    with open(pass_filename, "w") as f:
                        f.write(mlir_text)
                    print(f"Pass output written to: {pass_filename}")

        return mlir_text

    def verify_translate(self, mlir_text, output_dir=None, output_filename=None, output_info=True):
        """
        Run mlir-translate on the given MLIR text.
        Returns True if translation succeeds, False otherwise.
        """
        translate_cmd = [self.mlir_translate_path, "--verify-diagnostics", "--mlir-to-llvmir"]
        if output_info:
            print(f"Running: {' '.join(translate_cmd)}")
        translate_result = subprocess.run(translate_cmd, input=mlir_text, capture_output=True, text=True)
        if translate_result.returncode != 0:
            if output_info:
                print("mlir-translate verification failed:")
                print(translate_result.stderr)
            return False
        else:
            if output_info:
                print("mlir-translate verification succeeded.")

            if output_dir is not None:
                llvm_filename = output_filename if output_filename is not None else "output.ll"
                llvm_path = os.path.join(output_dir, llvm_filename)
                with open(llvm_path, "w") as f:
                    f.write(translate_result.stdout)
                if output_info:
                    print(f"LLVM IR written to: {llvm_path}")
            elif output_filename is not None:
                with open(output_filename, "w") as f:
                    f.write(translate_result.stdout)
                if output_info:
                    print(f"LLVM IR written to: {output_filename}")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".ll", mode="w") as temp_file:
                    temp_file.write(translate_result.stdout)
                    if output_info:
                        print(f"LLVM IR written to: {temp_file.name}")

        return True

    def verify(self, passes, output_dir=None, output_filename=None):
        """
        Full verification: runs passes then mlir-translate.
        Returns True if all succeed, False otherwise.
        """
        mlir_text = self.verify_passes(passes, output_dir=output_dir)
        if mlir_text is None:
            return False
        return self.verify_translate(mlir_text, output_dir=output_dir, output_filename=output_filename)

    def verify_passes_passman(self, passes):
        """
        Verify the passes using the PassManager.
        """
        pass_manager = PassManager(context=self.mlir_module.context)

        pass_manager.add(passes[-1])
        pass_manager.run(self.mlir_module.operation)
        return str(self.mlir_module)

class PassFuzzer:
    def __init__(self, mlir_module, passes, mlir_opt_path="mlir-opt", mlir_translate_path="mlir-translate"):
        """
        mlir_module: The MLIR module to test.
        passes: List or set of MLIR passes to try.
        """
        self.mlir_module = mlir_module
        self.passes = list(passes)
        self.mlir_opt_path = mlir_opt_path
        self.mlir_translate_path = mlir_translate_path

    def find_effective_pass_sequence(self, output_dir="fuzz_seq_results"):
        """
        Finds a sequence of passes where each pass changes the MLIR text,
        and the final result passes mlir-translate.
        Returns the sequence list.
        """
        verifier = MLIRVerifier(
            self.mlir_module,
            mlir_opt_path=self.mlir_opt_path,
            mlir_translate_path=self.mlir_translate_path
        )
        sequence = []
        current_mlir = str(self.mlir_module)
        available_passes = self.passes.copy()
        shutil.rmtree(output_dir) if os.path.exists(output_dir) and os.path.isdir(output_dir) else None
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, f"input_module.mlir"), "w") as f:
            f.write(current_mlir)
            print(f"Input module written to: {os.path.join(output_dir, f'input_module.mlir')}")

        while available_passes:
            progress = False
            for p in available_passes[:]:
                test_sequence = sequence + [p]
                print(f"Testing sequence: {test_sequence}")
                mlir_text = verifier.verify_passes(test_sequence,output_info=False)
                if mlir_text is not None and mlir_text.strip() != current_mlir.strip() and mlir_text != "":
                    sequence.append(p)
                    current_mlir = mlir_text
                    progress = True
                    with open(os.path.join(output_dir, f"step_{len(sequence)}_{p}.mlir"), "w") as f:
                        f.write(current_mlir)
                    if verifier.verify_translate(current_mlir, output_dir=output_dir, output_filename=f"finalized_llvm.ll", output_info=False):
                        print(f"Sequence found: {sequence}")
                        return sequence
                    break

            if not progress:
                break

        print("No valid sequence found that passes mlir-translate.")
        return

