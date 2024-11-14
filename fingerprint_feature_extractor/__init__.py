import cv2
import numpy as np
import skimage.morphology
from skimage.morphology import convex_hull_image, erosion
from skimage.morphology import square
import math

class MinutiaeFeature(object):
    def __init__(self, locX, locY, Orientation, Type):
        self.locX = locX
        self.locY = locY
        self.Orientation = Orientation
        self.Type = Type

class FingerprintFeatureExtractor(object):
    def __init__(self):
        self._mask = []
        self._skel = []
        self.minutiaeTerm = []
        self.minutiaeBif = []
        self._spuriousMinutiaeThresh = 10

    def setSpuriousMinutiaeThresh(self, spuriousMinutiaeThresh):
        self._spuriousMinutiaeThresh = spuriousMinutiaeThresh

    def __skeletonize(self, img):
        img = np.uint8(img > 128)
        self._skel = skimage.morphology.skeletonize(img)
        self._skel = np.uint8(self._skel) * 255
        self._mask = img * 255

    def _checkIsBifurcation(self, matrix):
        """
        Check if the center point is True and exactly 3 neighboring points are True
        in a symmetric matrix of odd size.
        
        Args:
            matrix: NumPy array containing boolean values
            
        Returns:
            bool: True if conditions are met, False otherwise
        """
        # Check if matrix is square and odd-sized
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
    
        if matrix.shape[0] != matrix.shape[1]:  # Check if matrix is square
            return False
        
        size = matrix.shape[0]
        if size % 2 == 0:  # Check if size is odd
            return False
            
        # Find center point
        center = size // 2
        
        # Check if center point is True
        if not matrix[center, center]:
            return False
            
        # Get neighboring points (up, down, left, right, diagonals)
        neighbors = [
            (center-1, center-1), (center-1, center), (center-1, center+1),
            (center, center-1),                       (center, center+1),
            (center+1, center-1), (center+1, center), (center+1, center+1)
        ]
        
        # Count True values in neighboring points
        true_count = sum(matrix[row, col] for row, col in neighbors)
        
        return true_count == 3

    def _get_unoccupied_neighbors(self, point, matrix, visited):
        """Returns unvisited neighbors with 'True' values, prioritizing horizontal and vertical."""
        i, j = point
        rows, cols = matrix.shape
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        neighbors = []
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if (
                0 <= ni < rows
                and 0 <= nj < cols
                and matrix[ni, nj]
                and (ni, nj) not in visited
            ):
                neighbors.append((ni, nj))
        return neighbors

    # Main function to perform the agent walk
    def _agent_walk(self, matrix):
        n = matrix.shape[0]
        center = (n // 2, n // 2)

        # Initialize agent paths
        agent_paths = {"A": [center], "B": [center], "C": [center]}
        visited = {center}  # Initially, the center is visited
        agent_labels = ["A", "B", "C"]

        # Each agent's initial move to unique neighbors
        neighbors = self._get_unoccupied_neighbors(center, matrix, visited)
        for i, agent in enumerate(agent_labels):
            if i < len(neighbors):  # Check if there are enough neighbors
                agent_paths[agent].append(neighbors[i])
                visited.add(neighbors[i])
                matrix[neighbors[i]] = False  # Mark as visited in matrix

        # Agents move until no more moves are available
        while True:
            moves_made = False  # Track if any move was made
            for agent in agent_labels:
                current_pos = agent_paths[agent][-1]
                unvisited_neighbors = self._get_unoccupied_neighbors(current_pos, matrix, visited)

                if unvisited_neighbors:
                    # Prioritize the first unvisited neighbor
                    next_move = unvisited_neighbors[0]
                    agent_paths[agent].append(next_move)
                    visited.add(next_move)
                    matrix[next_move] = False  # Mark as visited in matrix
                    moves_made = True

            if not moves_made:
                break  # Stop if no moves are possible for any agent

        # Convert the paths to Cartesian coordinates (x, y)
        rows = matrix.shape[0]
        cartesian_paths = {
            agent: [
                (col, rows - 1 - row) for row, col in path
            ]  # Swap and invert row for Cartesian
            for agent, path in agent_paths.items()
        }
        return cartesian_paths

    def _point_anchored_regression(self, x, y, vertical_threshold=0.0001):
        """
        Linear regression that always passes through the first point (x[0], y[0]).
        """
        x0, y0 = x[0], y[0]
        
        x_variance = np.var(x[1:])
        if x_variance < vertical_threshold:
            return x0, np.inf, True
        
        y_variance = np.var(y[1:])
        if y_variance / x_variance > 100:
            dx = x[1:] - x0
            dy = y[1:] - y0
            slope = np.mean(dx / dy)
            return x0 - y0 * slope, 1/slope, False
        
        dx = x[1:] - x0
        dy = y[1:] - y0
        slope = np.sum(dx * dy) / np.sum(dx * dx)
        intercept = y0 - slope * x0
        
        return intercept, slope, False

    def _plot_regression_with_direction(self, points):
        """
        Plot points and regression line with direction vector, starting from first point.
        """
        x = points[:, 0]
        y = points[:, 1]
        b_0, b_1, is_vertical = self._point_anchored_regression(x, y)
        
        if is_vertical:
            direction = np.array([0, 1 if y[-1] > y[0] else -1])
        else:
            x_range = np.linspace(min(x) - 0.5, max(x) + 0.5, 100)
            y_range = b_0 + b_1 * x_range
            direction = np.array([1, b_1])
            direction = direction / np.linalg.norm(direction)
            endpoint = np.array([x[-1], y[-1]])
            start_point = np.array([x[0], y[0]])
            if np.dot(direction, endpoint - start_point) < 0:
                direction = -direction
        
        
        return direction
    
    def _calculate_angle_to_x_axis(self, vector):
        """
        Calculate the counter-clockwise angle from positive x-axis to the vector.
        """
        x_axis = np.array([1, 0])
        dot_product = np.dot(vector, x_axis)
        angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        
        # Use cross product to determine if angle should be > 180 degrees
        cross_product = np.cross(x_axis, vector)
        if cross_product < 0:
            angle = 360 - angle
            
        return angle
    
    def _calculate_smallest_angle(self, v1, v2):
        """
        Calculate the smallest angle between two vectors (0-180 degrees).
        """
        dot_product = np.dot(v1, v2)
        angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        # Always return the smaller angle (<=180 degrees)
        return min(angle, 360 - angle)

    def _computeAngleBifurcation(self, block):
        cartesian_agent_paths = self._agent_walk(block)
        # print('cartesian_agent_paths type', type(cartesian_agent_paths['A']))
        # cartesian_agent_paths type <class 'list'>
        vector_A = self._plot_regression_with_direction(np.array(cartesian_agent_paths['A']))
        vector_B = self._plot_regression_with_direction(np.array(cartesian_agent_paths['B']))
        vector_C = self._plot_regression_with_direction(np.array(cartesian_agent_paths['C']))

        # Calculate smallest angles between pairs
        angles = {
            'A→B': self._calculate_smallest_angle(vector_A, vector_B),
            'B→C': self._calculate_smallest_angle(vector_B, vector_C),
            'C→A': self._calculate_smallest_angle(vector_C, vector_A)
        }

        # Find the pair with smallest angle
        smallest_angle_pair = min(angles.items(), key=lambda x: x[1])
        # print(f"Smallest angle is between {smallest_angle_pair[0]}: {smallest_angle_pair[1]:.1f}°")

        # Calculate angle of remaining line to x-axis
        remaining_line = None
        if smallest_angle_pair[0] == 'A→B':
            remaining_line = ('C', vector_C)
        elif smallest_angle_pair[0] == 'B→C':
            remaining_line = ('A', vector_A)
        else:
            remaining_line = ('B', vector_B)

        angle_to_x = self._calculate_angle_to_x_axis(remaining_line[1])
        correct_angle = (angle_to_x + 180) % 360
        flipped_angle = (360 - correct_angle) % 360
        # print('angle_to_x:', angle_to_x)
        # print('correct_angle:', correct_angle)
        # print(f"Angle of line {remaining_line[0]} to positive x-axis: {angle_to_x:.1f}°")

        return flipped_angle

    def __computeAngle(self, block, minutiaeType):
        angle = []
        (blkRows, blkCols) = np.shape(block)
        CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
        if (minutiaeType.lower() == 'termination'):
            sumVal = 0
            for i in range(blkRows):
                for j in range(blkCols):
                    if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                        realAngle = -math.degrees(math.atan2(i - CenterY, j - CenterX))
                        customAngle = (realAngle + 360) % 360
                        flippedAngle = (360 - customAngle) % 360
                        # print('realAngle:', realAngle)
                        # print('customAngle:', customAngle)
                        # print('flippedAngle:', flippedAngle)
                        # angle.append(customAngle)
                        angle.append(flippedAngle)
                        sumVal += 1
                        if (sumVal > 1):
                            angle.append(float('nan'))
            return (angle)

        elif (minutiaeType.lower() == 'bifurcation'):
            angleBifurcation = self._computeAngleBifurcation(block)
            angle.append(angleBifurcation)

            return (angle)    

    def __getTerminationBifurcation(self):
        self._skel = self._skel == 255
        (rows, cols) = self._skel.shape
        self.minutiaeTerm = np.zeros(self._skel.shape)
        self.minutiaeBif = np.zeros(self._skel.shape)

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if (self._skel[i][j] == 1):
                    block = self._skel[i - 1:i + 2, j - 1:j + 2]
                    block_val = np.sum(block)
                    if (block_val == 2):
                        self.minutiaeTerm[i, j] = 1
                    elif (block_val == 4):
                        self.minutiaeBif[i, j] = 1

        self._mask = convex_hull_image(self._mask > 0)
        self._mask = erosion(self._mask, square(5))  # Structuing element for mask erosion = square(5)
        self.minutiaeTerm = np.uint8(self._mask) * self.minutiaeTerm

    def __removeSpuriousMinutiae(self, minutiaeList, img):
        img = img * 0
        SpuriousMin = []
        numPoints = len(minutiaeList)
        D = np.zeros((numPoints, numPoints))
        for i in range(1,numPoints):
            for j in range(0, i):
                (X1,Y1) = minutiaeList[i]['centroid']
                (X2,Y2) = minutiaeList[j]['centroid']

                dist = np.sqrt((X2-X1)**2 + (Y2-Y1)**2)
                D[i][j] = dist
                if(dist < self._spuriousMinutiaeThresh):
                    SpuriousMin.append(i)
                    SpuriousMin.append(j)

        SpuriousMin = np.unique(SpuriousMin)
        for i in range(0,numPoints):
            if(not i in SpuriousMin):
                (X,Y) = np.int16(minutiaeList[i]['centroid'])
                img[X,Y] = 1

        img = np.uint8(img)
        return(img)

    def __cleanMinutiae(self, img):
        self.minutiaeTerm = skimage.measure.label(self.minutiaeTerm, connectivity=2)
        RP = skimage.measure.regionprops(self.minutiaeTerm)
        self.minutiaeTerm = self.__removeSpuriousMinutiae(RP, np.uint8(img))

    def __performFeatureExtraction(self):
        FeaturesTerm = []
        self.minutiaeTerm = skimage.measure.label(self.minutiaeTerm, connectivity=2)
        RP = skimage.measure.regionprops(np.uint8(self.minutiaeTerm))

        WindowSize = 2  # --> For Termination, the block size must can be 3x3, or 5x5. Hence the window selected is 1 or 2
        FeaturesTerm = []
        for num, i in enumerate(RP):
            (row, col) = np.int16(np.round(i['Centroid']))
            block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]
            angle = self.__computeAngle(block, 'Termination')
            if(len(angle) == 1):
                FeaturesTerm.append(MinutiaeFeature(col, row, angle, 1))
        # print('FeaturesTerm:', FeaturesTerm)

        FeaturesBif = []
        self.minutiaeBif = skimage.measure.label(self.minutiaeBif, connectivity=2)
        RP = skimage.measure.regionprops(np.uint8(self.minutiaeBif))
        WindowSize = 5  # --> For Bifurcation, the block size must be 3x3. Hence the window selected is 1
        for i in RP:
            (row, col) = np.int16(np.round(i['Centroid']))

            block = self._skel[row - WindowSize:row + WindowSize + 1, col - WindowSize:col + WindowSize + 1]

            isBifurcation = self._checkIsBifurcation(block)
            if (isBifurcation):
                angle = self.__computeAngle(block, 'Bifurcation')
                FeaturesBif.append(MinutiaeFeature(col, row, angle, 2))
        return (FeaturesTerm, FeaturesBif)

    def extractMinutiaeFeatures(self, img):
        self.__skeletonize(img)

        self.__getTerminationBifurcation()

        self.__cleanMinutiae(img)

        FeaturesTerm, FeaturesBif = self.__performFeatureExtraction()
        return(FeaturesTerm, FeaturesBif)

    def showResults(self, FeaturesTerm, FeaturesBif):
        
        (rows, cols) = self._skel.shape
        DispImg = np.zeros((rows, cols, 3), np.uint8)
        DispImg[:, :, 0] = 255*self._skel
        DispImg[:, :, 1] = 255*self._skel
        DispImg[:, :, 2] = 255*self._skel

        for idx, curr_minutiae in enumerate(FeaturesTerm):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))

        for idx, curr_minutiae in enumerate(FeaturesBif):
            row, col = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))
        
        cv2.imshow('output', DispImg)
        cv2.waitKey(0)

    def saveResult(self, img_name, FeaturesTerm, FeaturesBif):
        (rows, cols) = self._skel.shape
        DispImg = np.zeros((rows, cols, 3), np.uint8)
        DispImg[:, :, 0] = 255 * self._skel
        DispImg[:, :, 1] = 255 * self._skel
        DispImg[:, :, 2] = 255 * self._skel
        i = 0
        for idx, curr_minutiae in enumerate(FeaturesTerm):
            # print('curr_minutiae Term locX:', curr_minutiae.locX)
            # print('curr_minutiae Term locY:', curr_minutiae.locY)
            # print('curr_minutiae Term Orientation:', curr_minutiae.Orientation)
            col, row = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))

            # Calculate the endpoint of the line based on orientation
            line_length = 10
            original_angle=(360 - curr_minutiae.Orientation[0]) % 360
            custom_angle = ((360 - original_angle) * -1) if original_angle >  180 else original_angle
            # angle_rad = np.deg2rad(curr_minutiae.Orientation[0])  # Convert orientation to radians
            angle_rad = np.deg2rad(custom_angle)  # Convert orientation to radians

            # Calculate endpoint coordinates
            end_row = int(row - line_length * np.sin(angle_rad))
            end_col = int(col + line_length * np.cos(angle_rad))

            # Draw the line from center point to the calculated endpoint
            rr_line, cc_line = skimage.draw.line(row, col, end_row, end_col)
            skimage.draw.set_color(DispImg, (rr_line, cc_line), (0, 0, 255))
            

        for idx, curr_minutiae in enumerate(FeaturesBif):
            # print('curr_minutiae Bif, locX:', curr_minutiae.locX)
            # print('curr_minutiae Bif, locY:', curr_minutiae.locY)
            # print('curr_minutiae Bif, Orientation:', curr_minutiae.Orientation)
            col, row = curr_minutiae.locX, curr_minutiae.locY
            (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
            skimage.draw.set_color(DispImg, (rr, cc), (0, 255, 0))

            # Calculate the endpoint of the line based on orientation
            line_length = 10
            original_angle=(360 - curr_minutiae.Orientation[0]) % 360
            custom_angle = ((360 - original_angle) * -1) if original_angle >  180 else original_angle
            # angle_rad = np.deg2rad(curr_minutiae.Orientation[0])  # Convert orientation to radians
            angle_rad = np.deg2rad(custom_angle)  # Convert orientation to radians

            # Calculate endpoint coordinates
            end_row = int(row - line_length * np.sin(angle_rad))
            end_col = int(col + line_length * np.cos(angle_rad))

            # Draw the line from center point to the calculated endpoint
            rr_line, cc_line = skimage.draw.line(row, col, end_row, end_col)
            skimage.draw.set_color(DispImg, (rr_line, cc_line), (0, 255, 0))
            

        cv2.imwrite(f"./output/minutiae/{img_name.split('.')[0]}.jpg", DispImg)

def extract_minutiae_features(img_name, img, spuriousMinutiaeThresh=10, invertImage=False, showResult=False, saveResult=False):
    feature_extractor = FingerprintFeatureExtractor()
    feature_extractor.setSpuriousMinutiaeThresh(spuriousMinutiaeThresh)
    if (invertImage):
        img = 255 - img;

    FeaturesTerm, FeaturesBif = feature_extractor.extractMinutiaeFeatures(img)

    if (saveResult):
        feature_extractor.saveResult(img_name, FeaturesTerm, FeaturesBif)

    if(showResult):
        feature_extractor.showResults(FeaturesTerm, FeaturesBif)

    return(FeaturesTerm, FeaturesBif)
