package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = [
        "-parse_headers",
        "no_layering_check",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

py_binary(
   name = "tf_convnet_client",
   srcs = [
       "tf_convnet_client.py",
       "tf_convnet_inference_pb2.py"
   ],
   deps = [
       "@tf//tensorflow:tensorflow_py"
   ],
)

py_binary(
   name = "tf_convnet_export",
   srcs = [
       "tf_convnet_export.py",
   ],
   deps = [
       "@tf//tensorflow:tensorflow_py",
       "//tensorflow_serving/session_bundle:exporter",
   ],
)

serving_proto_library(
    name = "tf_convnet_inference_proto",
    srcs = ["tf_convnet_inference.proto"],
    has_services = 1,
    cc_api_version = 2,
    cc_grpc_version = 1,
)

cc_binary(
    name = "tf_convnet_inference",
    srcs = [
        "tf_convnet_inference.cc",
    ],
    linkopts = ["-lm"],
    deps = [
        "@grpc//:grpc++",
        "@tf//tensorflow/core:core_cpu",
        "@tf//tensorflow/core:framework",
        "@tf//tensorflow/core:lib",
        "@tf//tensorflow/core:protos_all_cc",
        "@tf//tensorflow/core:tensorflow",
        ":tf_convnet_inference_proto",
        "//tensorflow_serving/servables/tensorflow:session_bundle_config_proto",
        "//tensorflow_serving/servables/tensorflow:session_bundle_factory",
        "//tensorflow_serving/session_bundle",
        "//tensorflow_serving/session_bundle:manifest_proto",
        "//tensorflow_serving/session_bundle:signature",
    ],
)
