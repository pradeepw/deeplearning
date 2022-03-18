import numpy as np
from math import *
from numpy import array
from numpy import diag
from numpy import dot
from numpy import zeros
from numpy import linalg as la
from numpy.linalg import eig

#Create a square matrix with elements 61 to 69. 
arrange_op = np.arange(61,64)
print ("arrange_op \n", arrange_op)  
square_matrix = np.array([np.arange(61,64),np.arange(64,67),np.arange(67,70)])
print ("Square matrix \n", square_matrix)

#Create an upper and lower triangular matrix of size 6x6 with random values.
matrix = np.random.rand(6,6)
print("Original Matrix:\n",matrix)

# Upper triangular matrix
upper_triangular_matrix = np.triu(matrix)
print("upper triangular Matrix for original matrix:\n",upper_triangular_matrix)

# Lower triangular matrix
lower_triangular_matrix = np.tril(matrix)
print("lower triangular Matrix for original matrix:\n",lower_triangular_matrix)


#Create a diagonal matrix of size 3x3 with integer valules and do the following:
#i). Create a diagonal matrix with the values above the diagonal elements.
#ii). Create a diagonal matrix with the values below the diagonal elements.

M = np.array([np.arange(73,76),np.arange(32,35), np.arange(57,60)])
print("Matrix:\n",M)

#Create a 4x4 identity matrix with integer numbers.
id_matrix = np.identity(4)
print("Identity matrix:\n",id_matrix)

id_matrix_int = np.identity(4,dtype = int)
print("id_matrix_int :\n",id_matrix_int)

#Create a null matrix of integers with shape 6x6.
null_matrix = np.zeros(shape=(6,6))
print("null_matrix :\n",null_matrix)

# Create a matrix and find its transpose.
matrix_original = np.array([np.arange(0,3),np.arange(3,6)])
matrix_transpose = matrix_original.T

print("original matrix \n", matrix_original)
print("matrix_transpose matrix \n", matrix_transpose)
#Find the inverse of a matrix.
some_matrix = np.array([[1,2],[3,4]])
inverse_matrix = la.inv(some_matrix)
print("some_matrix \n",some_matrix)
print("inverse_matrix \n",inverse_matrix)

#Create 3x3 matrix and find the inverse of it.
A = np.array([[1,9,3],[4,5,6], [7,8,9]])
# Finding Inverse 
A_inv = la.inv(A)
print("A matrix \n",A)
print("A-1 matrix \n",A_inv)

#Create a matrix of size 3x3 and find the determinant of it.
A = np.array([[1,9,3],[4,5,6], [7,8,9]])
det_A = la.det(A)
print("determinant of A \n",det_A)

#Create a 4x4 matrix and slice the rows and columns to get the middle sub-matrix and find its determinant.
B = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print("print B \n",B)
# Slicing of rows and columns
sliced_matrix = B[1:3,1:3]
print("sliced matrix \n", sliced_matrix)
# Finding the determinant
det_slice_matrix = la.det(sliced_matrix)
print("det_slice_matrix \n",det_slice_matrix)

#Create matrix and add a scalar to it.
matrix1 = np.array([np.arange(27,32),np.arange(53,58),np.arange(33,38),np.arange(91,96)])
print("Original Matrix:\n",matrix1)
# Scalar Addition
Scalar_add = matrix1 + 1
print("resultant  Matrix after scalar add :\n",Scalar_add)

#Create two matrices as matrix1 & matrix2. Then, perform addition operation on them.

matrix2 = np.array([np.arange(27,32),np.arange(53,58),np.arange(33,38),np.arange(91,96)])
print("Original Matrix:\n",matrix1)
# matrix Addition
matrix_add = matrix1 + matrix2
print("resultant  Matrix after matrix add :\n",matrix_add)


# Create a matrix of size 3x6 and subtract a scalar from it.
# Creating matrices
matrix1 = np.array([np.arange(5,11), np.arange(47,53), np.arange(9,15)])
print("Matrix 1:\n", matrix1)

# Scalar Substraction
Scalar_sub = matrix1 - np.sin(np.pi/4)
print("Scalar Subtraction :\n",Scalar_sub)

# Create two matrices of size 2x3 and subtract second matrix from the first one.
matrix1 = np.array([np.arange(0,3),np.arange(3,6)])
matrix2 = np.array([np.arange(6,9),np.arange(9,12)])
print("Matrix 1:\n", matrix1)
print("Matrix 2:\n", matrix2)
# Matrix Subtraction
Matrix_subtraction = matrix1 - matrix2
print(Matrix_subtraction)

#Create a 4x3 matrix and divide it by a scalar.
matrix1 = np.array([np.arange(6,9),np.arange(14,17),np.arange(97,100),np.arange(54,57)])
print("Matrix 1:\n", matrix1)
# Scalar Division
Scalar_div = matrix1/2
print("Scalar Division: \n",Scalar_div)

#Create two matrices and divide the first matrix by the second one.
matrix1 = np.array([np.arange(18,21),np.arange(32,35)])
matrix2 = np.array([np.arange(6,9),np.arange(9,12)])
print("Matrix 1:\n", matrix1)
print("Matrix 2:\n", matrix2)
# Matrix Division
Matrix_div = matrix1/matrix2
print(Matrix_div)

#Create a matrix and mulitply it by a Scalar.
matrix1 = np.array([np.arange(121,126),np.arange(75,80)])
print("Matrix 1:\n", matrix1)

# Scalar Product
Scalar_product = matrix1*np.sin(3*np.pi/4)
print("Scalar Product:\n", Scalar_product)

# Create two matrices and multiply them.
matrix1 = np.array([np.arange(5,9),np.arange(2,6)])
matrix2 = np.array([np.arange(6,8),np.arange(12,14),np.arange(4,6),np.arange(1,3)])
print("Matrix 1:\n", matrix1)
print("Matrix 2:\n", matrix2)
Dot_product = np.dot(matrix1, matrix2)
print("Dot Product: \n",Dot_product)

#Create two matrices and perform cross product on them.
a = np.array([[4,8,9],[6,5,2]])
b = np.array([[1,4,8],[3,7,5]])
print("Matrix a\n",a)
print("Matrix b\n",b)

# Cross Product
cproduct = np.cross(a,b)
print("Cross Product of Matrices:\n",cproduct)



#    Apply dot product on the below given vectors:
##   i). x = sin 45 + 6i
##    ii). y = 42 + 7i
y = [42,7]
x = [sin(np.pi/4),6]
dot_pdct = np.dot(x,y)
print("dot_pdct \n", dot_pdct)


#Create two vectors and perform cross product operation on them.
a = [9,6,3]
b = [2,5,9]
# Cross Product
cross_product = np.cross(a,b)
print("Cross Product of Vectors:\n",cross_product)

#Solve the following linear equation for x and y using matrix solution.
## i). 7x + 3y = 34
## ii). 8x + 9y = 50
# Creating matrices
A = np.array([[7,3],[8,9]])
X = np.array([[x],[y]])
B = np.array([[34],[50]])
# Inverse of matrix A
A_inv = la.inv(A)
# Dot product of matrix A_inv and B
X = np.dot(A_inv,B)
print(" values of X \n",X)



#Find the values of x and y from the system of linear equations given below and verify the solution:
##   i). 6x + 5y = 19
##  ii). 7x + 9y = 32
# Creating arrays
# YOUR CODE HERE
A = np.array([[6,5],[7,9]])
X = np.array([[x],[y]])
B = np.array([[19],[32]])
# Solve a linear matrix equation
values = la.solve(A,B)
print(" values of X \n",values)
# Verify the solution

# Creating matrices
p = np.dot(A,values)
result = np.allclose(p,B)

print("Result:\n",result)



#    Below are pair of three linear equations. Find the values of x, y, & z.
##    i). 8x + 3y + 5z = 29
##    ii). 4x + 7y + 12z = 54
##    iii). 9x + 5y + 11z = 52

# Creating matrices
A = np.array([[8,3,5],[4,7,12],[9,5,11]])
B = np.array([[29],[54],[52]])
# Inverse of matrix A
A_inv = la.inv(A)
# Dot product of A1 and B
X = np.dot(A_inv,B)
print("values \n",X)


#Suppose an automobile company sells 25 bikes and 32 scooties in January. In the next month the sale goes up to 30 bikes and 41 scooties. The total revenue generated is Rs. 506,000 and Rs.620,000 for the respective months. Find the cost of bike and scooty using matrix solution.
# Create matrices using the values from the equation
A = np.array([[25,32],[30,41]])
B = np.array([[506000],[628000]])
# Inverse of matrix A 
A_inv = la.inv(A)
# Dot product of A1 and B
X = np.dot(A_inv,B)
print("values \n",X)



# Find whether the below linear equations are linearly dependent or independent.
##    i). 4x + 2y = 6
##   ii). y = -x-2
a = np.array([[4,2],[1,1]])
print(a)
# Finding determinant
a_det = la.det(a)
# Applying the condition
print("determinant \n", a_det)
if (a_det == 0):
  print(" These are linearly dependent as determinant is 0")
else :
  print(" These are linearly independent as determinant is not 0")

# Create a matrix and find the eigen vectors and eigen values for it.
# Create a matrix
A = np.array([[2,6],[3,5]])
print("Array:\n",A)
# Calculating eigen vectors and eigen values
val,vec = la.eig(A)

print("Eigen Value:\n",val)
print("Eigen Vector:\n",vec)

# Confirm that the vector for the  matrix is an eigen vector of the matrix.
# Hint: Equate the multiplication of first eigen vector with multiplication of first eigen vector and first eigen value.

# Defining a matrix
M = np.array([[3,2,7],[5,8,9],[9,3,7]])
# Finding eigen vectors and values
val,vec = eig(M)
print("eigen values \n", val)
print("eigen vector \n", vec)
# Finding first vector
N = np.dot(M,vec[:,0])
print("First Vector:\n", N)
# Finding second vector
P = vec[:,0]*val[0]
print("Second Vector:\n",P)




# Create a matrix and try to reconstruct the original matrix using the eigen values and vectors.
##
### Steps to reconstruct the matrix using eigen values and vectors:
###   i). Create inverse of eigen vector matrix
###   ii). Create diagonal matrix from eigen values
###     iii). Get the original matrix by dot product of eigen vector matrix, diagonal matrix, and inverse of eigen vector matrix.
# Defining matrix
M = np.array([[6, 8, 5], [4, 7, 1], [9, 2, 6]])
print("Initial Matrix:\n",M)
val, vec = eig(M)
print("eigen values \n", val)
print("eigen vector \n", vec)
# Create matrix from eigenvectors
# Create inverse of eigenvectors matrix
vec_inv = la.inv(vec)
# Create diagonal matrix from eigenvalues
val_diag = np.diag(val)
# Reconstruct the original matrix. Note: Here the dot product is not communtative
reconstructed_matrix = vec.dot(val_diag).dot(vec_inv)
print("reconstructed_matrix \n",reconstructed_matrix)


#
Find the sum of eigen values of the below matrix:
[[2,7,6],[3,5,9],[7,9,11]]