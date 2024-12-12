import argparse
import csv

def process_log_files(log_files):
    for log_file in log_files:
        
        
        results = []
        with open(log_file, 'r') as file:
            lines = file.readlines()

        log_type = 1
        for i in range(len(lines)):
            if '@5' in lines[i]:
                log_type = 2
                break
        if log_type == 2:
            for i in range(len(lines)):
                if lines[i].startswith(' *  Acc'):
                    acc_line = lines[i].strip()
                    acc_values = acc_line.split(' ')
                    # print(acc_values)
                    acc1 = acc_values[-3]  # Acc@1
                    acc5 = acc_values[-1]  # Acc@5
                    
                    testset_line = lines[i + 1].strip()
                    testset_name = testset_line.split('[')[1].split(']')[0]

                    results.append([testset_name, acc1, acc5])

            csv_file = log_file[: -4] + '.csv'
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(['Testset', 'Acc@1', 'Acc@5'])
                # 写入内容
                writer.writerows(results)

            print(f"Data has been written to {csv_file}")

        else:
            for i in range(len(lines)):
                if "number of test samples" in lines[i] and "Acc. on testset" in lines[i + 1]:
                    # 提取测试集编号、样本数和准确率
                    num_samples = lines[i].split(": ")[1].strip()
                    testset_info = lines[i + 1].split("[")[1].split("]")[0]
                    accuracy = lines[i + 1].split(": ")[1].strip()
                    results.append([testset_info, num_samples, accuracy])


            csv_file = log_file[: -4] + '.csv'

            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(["Testset", "Number of Samples", "Accuracy"])
                # 写入内容
                writer.writerows(results)

            print(f"Data has been written to {csv_file}")

def main():
    parser = argparse.ArgumentParser(description='Process multiple log files and save results to a CSV.')
    parser.add_argument('log_files', type=str, nargs='+', help='Paths to the log files (multiple files allowed)')
    args = parser.parse_args()

    process_log_files(args.log_files)

if __name__ == '__main__':
    main()