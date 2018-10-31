#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using namespace std;

int max_pt_num = 2048;


// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}


Status GetTopLabels(const std::vector<Tensor>& outputs, int how_many_labels,
                            Tensor* indices, Tensor* scores) 
{
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string output_name = "top_k";
  TopK(root.WithOpName(output_name), outputs[0], how_many_labels);
  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensors.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));
  
  std::unique_ptr<tensorflow::Session> session(
  tensorflow::NewSession(tensorflow::SessionOptions()));
  TF_RETURN_IF_ERROR(session->Create(graph));
  // The TopK node returns two outputs, the scores and their original indices,
  // so we have to append :0 and :1 to specify them both.
  std::vector<Tensor> out_tensors;
  TF_RETURN_IF_ERROR(session->Run({}, {output_name + ":0", output_name + ":1"}, {}, &out_tensors));
  *scores = out_tensors[0];
  *indices = out_tensors[1];
  return Status::OK();
}



int main(int argc, char* argv[]) {
    string input_layer = "inputPointcloud";
    string input_flag = "isTraining";
    string output_layer = "output/prob";
  
    // First we load and initialize the model.
    std::unique_ptr<tensorflow::Session> session;
    string graph_path = "../model/model.pb";
    Status load_graph_status = LoadGraph(graph_path, &session);
    if (!load_graph_status.ok()) {
      LOG(ERROR) << load_graph_status;
      return -1;
    }
  
    string pcd_path = "../test_van.pcd";
    
    // cout << "Starting loading data ......" << endl;
    /****************** PCL part **********************/
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile <pcl::PointXYZ>(pcd_path, *cloud);
    
    // cout << "Filtering ......" << endl;
    // Filter oversized pointcloud
    float leaf = 0.01;
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    
    while(cloud->size() > max_pt_num)
    {
        // cout << "Point num before filtering: " << (cloud->size()) << endl;
        
        sor.setLeafSize(leaf, leaf, leaf);
        sor.setInputCloud(cloud);
        sor.filter(*cloud_filtered);
        cloud.swap(cloud_filtered);
        leaf += 0.01f;

        // cout << "Point num after filtering: " << (cloud->size()) << endl;
    }
    // cout << "PCL filtering successful!" << endl;
    /******************* end of PCL part **************/

    Tensor data_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, max_pt_num, 3}));
    auto data_mapped = data_tensor.tensor<float, 3>();

    for(unsigned int i = 0; i < cloud->points.size(); ++i)
    {
        data_mapped(0, i, 0) = cloud->points[i].x;
        data_mapped(0, i, 1) = cloud->points[i].y;
        data_mapped(0, i, 2) = cloud->points[i].z;
    }
    for(unsigned int j = cloud->points.size(); j < max_pt_num; ++j)
    {
        for(int k = 0; k < 3; ++k)
        {
            data_mapped(0, j, k) = 0;
        }
    }

    // cout << "Passing tensor ......" << endl;

    // cout << "Got data_tensor!" << endl;

    tensorflow::Tensor phase_tensor(tensorflow::DT_BOOL, tensorflow::TensorShape());
    phase_tensor.scalar<bool>()() = false;
    
    // cout << "Set phase_tensor ......" << endl;
    
    vector<pair<string, Tensor> > inputs = {
        {input_layer, data_tensor},
        {input_flag, phase_tensor},
    };

    // cout << "Set inputs" << endl;
   
    // Actually run the image through the model.
    std::vector<Tensor> outputs;

    // cout << "Before session run" << endl;

    Status run_status = session->Run(inputs,
                                     {output_layer}, {}, &outputs);
    if (!run_status.ok()) {
      LOG(ERROR) << "Running model failed: " << run_status;
      return -1;
    }
 
    cout << "Runing model successful!" << endl;
    
    // std::cout<< outputs[0].matrix<float>() << std::endl;

    Tensor indices;
    Tensor scores;

    Status predict_status = GetTopLabels(outputs, 1, &indices, &scores);
    if(!predict_status.ok())
    {
        LOG(ERROR) << "Running print failed: " << predict_status;
        return -1;
    }
    tensorflow::TTypes<int32>::Flat indices_flat = indices.flat<int32>();
    tensorflow::TTypes<float>::Flat scores_flat = scores.flat<float>();

    const int label = indices_flat(0);
    const float prob = scores_flat(0);

    cout << "Label index: " << label << " with probability of " << prob << endl;
  
  return 0;
}
