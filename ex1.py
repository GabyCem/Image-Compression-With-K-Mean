import numpy
import numpy as np
import sys
import matplotlib.pyplot as plt

image_fname, centroids_fname, out_fname = sys.argv[1], sys.argv[2], sys.argv[3]
z = np.loadtxt(centroids_fname)  # load centroids
orig_pixels = plt.imread(image_fname)
pixels = orig_pixels.astype(float) / 255.
# Reshape the image(128x128x3) into an Nx3 matrix where N = number of pixels
pixels = pixels.reshape(-1, 3)
iterations_vector = []
avg_loss_vector = []


def cleanClusters(clusters):
    for i in range(len(clusters)):
        clusters[i] = []


# This function compares to list of lists.
def listCompare(l1, l2):
    if len(l1) != len(l2):
        return False
    for i in range(len(l1)):
        if not np.array_equal(l1[i], l2[i]): return False

    return True


def kMean():
    # open the file
    f = open(sys.argv[3], "w+")
    count = 0
    centroid_change_flag = 0
    k = len(z)
    clusters = list()
    prev_centroids = list(z)

    for i in range(k):
        clusters.insert(i, [])

    while count != 20 and centroid_change_flag != 1:
        min_distance = 9999999999
        min_index = 0
        pixel_index = 0
        total_loss = 0
        # iterate over the pixels
        for pixel in pixels:
            pixel_index += 1
            centroid_index = 0
            # iterate over the centroids
            for centroid in prev_centroids:
                # calculate the distance.
                cur_distance = np.sqrt((pixel[0] - centroid[0]) ** 2 + (pixel[1] - centroid[1]) ** 2
                                       + (pixel[2] - centroid[2]) ** 2)
                if cur_distance < min_distance:
                    min_distance = cur_distance
                    min_index = centroid_index
                centroid_index += 1
            # choose correct cluster with min distance.
            clusters[min_index].append(pixel)
            total_loss += min_distance ** 2
            min_distance = 9999999999

        # calc the loss for the plots.
        avg_loss = total_loss / len(pixels)
        iterations_vector.append(count)
        avg_loss_vector.append(avg_loss)
        cluster_index = 0
        updated_centroids = list()
        for cluster in clusters:
            x_total_val = 0
            y_total_val = 0
            z_total_val = 0
            for i in range(len(cluster)):
                x_total_val += cluster[i][0]
                y_total_val += cluster[i][1]
                z_total_val += cluster[i][2]
            if len(cluster) != 0:
                average_x = x_total_val / (len(cluster))
                average_y = y_total_val / (len(cluster))
                average_z = z_total_val / (len(cluster))
            else:
                average_x = 0
                average_y = 0
                average_z = 0
            new_centroid = np.array([average_x, average_y, average_z])
            updated_centroids.insert(cluster_index, (new_centroid.round(4)))
            cluster_index += 1

        if listCompare(prev_centroids, updated_centroids):
            centroid_change_flag = 1

        prev_centroids = updated_centroids

        f.write(f"[iter {count}]:{','.join([str(i) for i in updated_centroids])}\n")
        count += 1
        cleanClusters(clusters)
    f.close()
    return iterations_vector, avg_loss_vector


def main():
    x,y = kMean()
    # plotting the points
    plt.plot(x, y)
    # naming the x axis
    plt.xlabel('iterations')
    # naming the y axis
    plt.ylabel('avg/total loss')

    # giving a title to my graph
    plt.title('k-mean analysis: K = ' + str(len(z)))

    # function to show the plot
    plt.show()
    return 0


if __name__ == "__main__":
    main()
