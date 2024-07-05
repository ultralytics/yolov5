# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Utils to interact with the Triton Inference Server."""

import typing
from urllib.parse import urlparse

import torch


class TritonRemoteModel:
    """
    A wrapper over a model served by the Triton Inference Server.

    It can be configured to communicate over GRPC or HTTP. It accepts Torch Tensors as input and returns them as
    outputs.
    """

    def __init__(self, url: str):
        """
        Initializes the TritonRemoteModel instance to facilitate interaction with the Triton Inference Server.

        Args:
            url (str): Fully qualified address of the Triton server, e.g., "grpc://localhost:8000".

        Returns:
            None

        Raises:
            ValueError: If the URL scheme is not recognized (neither "grpc" nor standard HTTP).

        Example:
            ```python
            from ultralytics import TritonRemoteModel

            # Initialize the model with a GRPC URL
            model = TritonRemoteModel(url="grpc://localhost:8000")
            ```

        Notes:
            The function dynamically imports Triton client modules based on the protocol used (GRPC or HTTP).
        """

        parsed_url = urlparse(url)
        if parsed_url.scheme == "grpc":
            from tritonclient.grpc import InferenceServerClient, InferInput

            self.client = InferenceServerClient(parsed_url.netloc)  # Triton GRPC client
            model_repository = self.client.get_model_repository_index()
            self.model_name = model_repository.models[0].name
            self.metadata = self.client.get_model_metadata(self.model_name, as_json=True)

            def create_input_placeholders() -> typing.List[InferInput]:
                return [
                    InferInput(i["name"], [int(s) for s in i["shape"]], i["datatype"]) for i in self.metadata["inputs"]
                ]

        else:
            from tritonclient.http import InferenceServerClient, InferInput

            self.client = InferenceServerClient(parsed_url.netloc)  # Triton HTTP client
            model_repository = self.client.get_model_repository_index()
            self.model_name = model_repository[0]["name"]
            self.metadata = self.client.get_model_metadata(self.model_name)

            def create_input_placeholders() -> typing.List[InferInput]:
                return [
                    InferInput(i["name"], [int(s) for s in i["shape"]], i["datatype"]) for i in self.metadata["inputs"]
                ]

        self._create_input_placeholders_fn = create_input_placeholders

    @property
    def runtime(self):
        """
        Returns the model runtime.

        Returns:
            str: Runtime used by the model, either 'grpc' or 'http'.

        Examples:
            ```python
            triton_model = TritonRemoteModel(url='grpc://localhost:8000')
            runtime = triton_model.runtime  # 'grpc'
            ```
        """
        return self.metadata.get("backend", self.metadata.get("platform"))

    def __call__(self, *args, **kwargs) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, ...]]:
        """
        Invokes the model hosted on the Triton Inference Server.

        Args:
            *args (torch.Tensor): Positional arguments matching the order of the model inputs.
            **kwargs (torch.Tensor): Keyword arguments where keys match the model input names.

        Returns:
            torch.Tensor | tuple[torch.Tensor, ...]: The output tensor(s) from the model inference.

        Notes:
            - Ensure the input tensors are appropriately ordered or named to match the model's expectation.
            - The Triton Inference Server device must be reachable and properly configured.

        Examples:
            ```python
            model = TritonRemoteModel(url='grpc://localhost:8000')
            input_tensor = torch.randn(1, 3, 224, 224)
            output_tensor = model(input_tensor)
            ```
            ```python
            model = TritonRemoteModel(url='http://localhost:8000')
            input_tensor1 = torch.randn(1, 3, 224, 224)
            input_tensor2 = torch.randn(1, 5)
            output_tensor1, output_tensor2 = model(input1=input_tensor1, input2=input_tensor2)
            ```
        """
        inputs = self._create_inputs(*args, **kwargs)
        response = self.client.infer(model_name=self.model_name, inputs=inputs)
        result = []
        for output in self.metadata["outputs"]:
            tensor = torch.as_tensor(response.as_numpy(output["name"]))
            result.append(tensor)
        return result[0] if len(result) == 1 else result

    def _create_inputs(self, *args, **kwargs):
        """
        Creates input tensors from given arguments or keyword arguments.

        Args:
            *args: Variable length argument list representing the input tensors in the order expected by the model.
            **kwargs: Arbitrary keyword arguments representing input tensors keyed by input names.

        Returns:
            typing.List[InferInput]: A list of Triton InferInput objects containing input data derived from the provided
            arguments.

        Raises:
            RuntimeError: If no inputs are provided, or if both `args` and `kwargs` are provided simultaneously. Also raises an
            error if the number of provided `args` does not match the expected number of inputs.

        Examples:
            ```python
            # Using positional arguments
            inputs = model._create_inputs(tensor1, tensor2)

            # Using keyword arguments
            inputs = model._create_inputs(input_name1=tensor1, input_name2=tensor2)
            ```

        Notes:
            This function is designed to support interfacing with Triton Inference Server, ensuring the input tensors are
            properly formatted and match the server's expected inputs.
        """
        args_len, kwargs_len = len(args), len(kwargs)
        if not args_len and not kwargs_len:
            raise RuntimeError("No inputs provided.")
        if args_len and kwargs_len:
            raise RuntimeError("Cannot specify args and kwargs at the same time")

        placeholders = self._create_input_placeholders_fn()
        if args_len:
            if args_len != len(placeholders):
                raise RuntimeError(f"Expected {len(placeholders)} inputs, got {args_len}.")
            for input, value in zip(placeholders, args):
                input.set_data_from_numpy(value.cpu().numpy())
        else:
            for input in placeholders:
                value = kwargs[input.name]
                input.set_data_from_numpy(value.cpu().numpy())
        return placeholders
