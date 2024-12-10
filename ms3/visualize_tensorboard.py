import os
import re
import csv
import tensorflow as tf
from tensorflow.core.util import event_pb2

def tensorboard_logs_to_csv(log_dir, regex_pattern, output_dir):
    pattern = re.compile(regex_pattern)
    tags_to_extract = ['epoch_age_age_acc', 'epoch_face_accuracy', 'epoch_gender_gen_acc']
    data = {tag: [] for tag in tags_to_extract}

    for root, dirs, files in os.walk(log_dir):
        if pattern.search(root):
            event_files = [os.path.join(root, f) for f in files if 'events.out.tfevents' in f]
            for event_file in event_files:
                try:
                    for raw_record in tf.data.TFRecordDataset(event_file):
                        event = event_pb2.Event()
                        event.ParseFromString(raw_record.numpy())
                        for value in event.summary.value:
                            if value.tag in tags_to_extract:
                                # Check for simple_value
                                if value.HasField('simple_value'):
                                    scalar_value = value.simple_value
                                # Check for tensor value
                                elif value.HasField('tensor'):
                                    tensor_proto = value.tensor
                                    tensor_value = tf.make_ndarray(tensor_proto)
                                    scalar_value = tensor_value.item()
                                else:
                                    continue  # Skip if no scalar value is found
                                
                                # Append the data
                                data[value.tag].append({
                                    'run': os.path.relpath(root, log_dir),
                                    'step': event.step,
                                    'value': scalar_value
                                })
                except Exception as e:
                    print(f"Error processing {event_file}: {e}")
                    continue

    # Write each tag's data to a separate CSV file
    os.makedirs(output_dir, exist_ok=True)
    for tag in tags_to_extract:
        records = data[tag]
        output_csv = os.path.join(output_dir, f"{tag}.csv")
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['run', 'step', 'value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
            writer.writeheader()
            for row in records:
                writer.writerow(row)
        print(f"Written {len(records)} records to {output_csv}")

if __name__ == "__main__":
    tensorboard_logs_to_csv('logs/', r'ms3_final.*validation$', 'output_csvs')