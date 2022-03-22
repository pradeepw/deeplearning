import numpy as np
import scipy
import sympy as sym
import matplotlib.pyplot as plt
from sympy.abc import x, y,z
from scipy import linalg

#Rank of a matrix: - The rank of a matrix is defined as the maximum number of linearly independent column vectors in the matrix or the maximum number of linearly independent row vectors in the matrix.
#Full Row Rank:- If the rank of a matrix is equal to the number of row in the matrix, then matrix has a Full Row Rank.
#Full Column Rank:- If the rank of a matrix is equal to the number of columns in the matrix, then matrix has Full Column Rank.
#Rank Deficient:-When the rank of a matrix is less than the number of rows and columns in the matrix, then the matrix is Rank Deficient.
#Uses:- It is useful in understanding whether we have a chance of solving a system of linear equations or not. Further, if rank is equal to the number of variables we will be able to find a unique solution
#Create a random matrix and find the rank of it.

matrix_a = np.random.randint(10, size= (2, 4)) 
print("Matrix:", matrix_a)

rank_a = np.linalg.matrix_rank(matrix_a)
print("Matrix rank:", rank_a)

#The Singular-Value Decomposition, or SVD is a matrix decomposition method for reducing a matrix to its constituent parts in order to make certain subsequent matrix calculations simpler. The singular value decomposition (SVD) provides another way to factorize a matrix, into singular vectors and singular values. The SVD allows us to discover some of the same kind of information as the eigendecomposition.
#Steps to follow for Singular Value Decomposition
#A. Factorize a matrix using SVD
#B. Reconstruct the original matrix
#A. Matrix Factorization:

#In simple terms, SVD is the factorization of a matrix into 3 matrices. So if we have a matrix A, then its SVD is represented by:
#𝐴=𝑈∑𝑉𝑇
#Where A is an 𝑚𝑥𝑛 matrix,
#    𝑈 is an 𝑚𝑥𝑚 orthogonal matrix
#    ∑ is an 𝑚𝑥𝑛 diagonal matrix with non-negative real numbers.
#    𝑉 is an 𝑛𝑥𝑛 orthogonal matrix.

#𝑈 is also referred to as the left singular vectors, ∑ the singular values, and 𝑉 the right singular vectors.
#B. Reconstruct the original matrix from SVD:
#The original matrix can be reconstructed from the 𝑈, Sigma, and 𝑉𝑇 elements.

#Create an array of 2x3 matrix and apply singular value decompostition on it.

# Define a matrix
A = np.array([[3, 1, 1], [-1, 3, 1]])
print("Matrix A:\n", A)

# Singular Value Decomposition
U, s, VT = np.linalg.svd(A)

# Create m x n matrix called sigma
sigma = np.zeros((A.shape[0], A.shape[1]))

# Populate sigma with n x n diagonal matrix
sigma[:A.shape[0], :A.shape[0]] = np.diag(s)
print("Matrix Sigma:\n", sigma)

# Reconstruct the original matrix
B = U.dot(sigma.dot(VT))

print("Matrix U\n", U)
print("Matrix s\n", s)
print("Matrix VT\n", VT)
print("Reconstructed Matrix B\n", B)

#Pseudo Inverse of Matrix
#Pseudo inverse of a matrix is used:
#    To solve Linear Equations using the Moore-Penrose Pseudoinverse.
#    For overdetermined systems (where an equation has no perfect solution but by using this we can get approximate solution)
#    To find least square solution in the triangle center.
#    To rearrange the original data from the decomposed data.

#Solve the matrix 𝐴=⎡⎣⎢⎢⎢⎢⎢786895⎤⎦⎥⎥⎥⎥⎥ by the pseudo inverse formula and verify the result.
# Creating the matrix
A = np.array([[7, 8], [8, 9], [6, 5]]) 

# Implementing the Singular Value Decomposition
U, Sigma, VT = np.linalg.svd(A) 

# Creating a matrix of shape transpose(matrix A)
# For square matrices inverse can be calculated directly
# For rectangular matrices psuedo inverse is way of finding inverse 
# so we need to use transpose when calculating for inverse of rectangular matrices
Sigma_plus = np.zeros((A.shape[0], A.shape[1])).T 

# Creating the inverse of matrix Sigma
Sigma_plus[:Sigma.shape[0], :Sigma.shape[0]] = np.linalg.inv(np.diag(Sigma)) 

# Creating a pseudo inverse A 
A_pseudo_inverse = VT.T.dot(Sigma_plus).dot(U.T) 

print("The pseudo inverse of matrix A by calculating through formula is\n", A_pseudo_inverse)
print("The pseudo inverse of matrix A by function is \n", np.linalg.pinv(A))

if np.all(np.linalg.pinv(A)) == np.all(A_pseudo_inverse): 
  print("Your procedure is correct to find the pseudo inverse matrix")
else:
  print("Something went wrong in the calculation")

#Solve the following equation by using Pseudo inverse method:
# 2x + y =2
# 4x - y = -8
# x + y = 2
# Hint: To solve the equation we need to follow the linear equation method i.e. 𝐴𝑥=𝑦.
# According to the above equations our matrices would be:
# 𝐴=⎡⎣⎢⎢⎢⎢⎢2411−11⎤⎦⎥⎥⎥⎥⎥𝑥=⎡⎣⎢⎢𝑥1𝑥2⎤⎦⎥⎥𝑦=⎡⎣⎢⎢⎢⎢⎢2−82⎤⎦⎥⎥⎥⎥⎥

# Let's see graphical representation of the equations 
x = np.linspace(-5, 5, 1000)

y_1 = -2 * x + 2 
y_2 = 4 * x + 8
y_3 = -1 * x + 2

#We actually see that there is no solution, as there is no point at the intersection of the three lines corresponding to three equations. Moreover, by using the pseudo-inverse method, we can get a pseudo solution. The solution obtained is an approximate solution but not an exact solution.
#We will now calculate the pseudoinverse of A
# Defining a matrix
A = np.array([[2, 1], [4, -1], [1, 1]]) 

# Finding the pseudo inverse
A_pseudo_inverse = np.linalg.pinv(A) 
print(A_pseudo_inverse) 
# we can use it to find x knowing that:
# 𝑥=𝐴+𝑦 with 𝑥=⎡⎣⎢⎢𝑥1𝑥2⎤⎦⎥⎥

# Creating the matrix y
y = np.array([[2], [-8], [2]])

# If we multiply pseudo inverse of A and y matrix we get a pseudo solution which might not be an exact solution 
result = A_pseudo_inverse.dot(y) 
print(result)

#In our two dimensions space the above result is the coordinates of x.
#Let’s plot this point along with the equations lines:
# Plotting the equation and the result
plt.plot(x, y_1, "-b", label="y_1")
plt.plot(x, y_2, "-r", label="y_2")
plt.plot(x, y_3, "-g", label="y_3")
plt.legend(loc="upper right")
plt.xlim(-2., 1)
plt.ylim(1, 5)
plt.scatter(result[0], result[1])
plt.show()

#Matrix Norm: The norm of a matrix is a measure of how large its elements are. It is a way of determining the “size” of a matrix that is not necessarily related to how many rows or columns the matrix has.
#The norm of a square matrix A is a non-negative real number denoted ‖𝐴‖. There are several different ways of defining a matrix norm, but they all share the following properties:
#    ‖𝐴‖ ≥ 0 for any square matrix A.
#    ‖𝐴‖ = 0 if and only if the matrix A = 0.
#    ‖𝑘𝐴‖ = |k| ‖𝐴‖, for any scalar k.
#    ‖𝐴+𝐵‖ ≤ ‖𝐴‖ + ‖𝐵‖.
#    ‖𝐴𝐵‖ ≤ ‖𝐴‖ ‖𝐵‖.
#Below is the formula to calculate 1 norm of a matrix:
#‖𝐴‖1=max1≤𝑗≤𝑛(∑𝑛𝑖∣∣𝑎𝑖𝑗∣∣)
#By using this formula we compute the sum of absolute values down each column and then take the largest number.

#Calculate the 1-norm of a matrix 𝐴=⎛⎝⎜⎜5−1−2−421230⎞⎠⎟⎟. 

# Creating a matrix
A = np.array([[ 5, -4, 2],[-1, 2, 3],[-2, 1, 0]])

# 1-Norm of a matrix
# The 1-Norm of a matrix is the maximum of the sum of each column
# Pass the matrix to the norm function and specify the order
matrix_norm = np.linalg.norm(A, 1)
print("Norm of a Matrix A is:",matrix_norm)

#Vector Norm: The norm of a vector in vector space is a real non-negative value representing intuitively the length, size, or magnitude of the vector.
#The most commonly used vector norms belong to the family of 𝑝-norms, or 𝑙𝑝-norms, which are defined by:
#‖𝑥‖𝑝=(∑𝑛𝑖=1|𝑥𝑖|𝑝)1/𝑝
#It can be shown that for any 𝑝>0, ∥⋅∥𝑝 defines a vector norm. The following 𝑝-norms are of particular interest:

#    𝑝=1: The 𝑙1-norm
#    ∥𝑥∥1=|𝑥1|+|𝑥2|+....+|𝑥𝑛|

#    𝑝=2: The 𝑙2-norm or Euclidean norm
#    ‖𝑥‖2=𝑥21+𝑥22+...+𝑥2𝑛‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾√

#    𝑝=∞: The 𝑙∞-norm
#    ‖𝑥‖∞=max1≤𝑖≤𝑛|𝑥𝑖|.

#Compute the norm of a given vector 𝐴=[5,12]
# Creating a matrix
vector = np.array([5,12])

# L2-Norm of a vector
vector_norm = np.linalg.norm(vector, 2)
print("Vector norm:",vector_norm)

#Differentiate the below given scalar function 𝑓(𝑥) w.r.t. to a scalar 𝑥.
# 𝑓(𝑥)=cos𝑥+sin𝑥+𝑥2
# Declaring symbols that can be used as variables in the function
x = sym.Symbol('x')

# Defining a function 'f'
f = sym.Function('f')
f = sym.cos(x) + sym.sin(x) + x * x

# Derivative of the scalar function w.r.t. a scalar x using the 'diff' function of Sympy
derivative_f = sym.diff(f, x)
print("Derivative of the function is:", derivative_f)

#Differentiate the given vector 𝑦⃗ =⎛⎝⎜⎜⎜𝑥2+2𝑥𝑠𝑖𝑛𝑥𝑐𝑜𝑠𝑥⎞⎠⎟⎟⎟ w.r.t. a scalar 𝑥.
y = sym.Matrix([[x * x + 2 * x],[sym.sin(x)], [sym.cos(x)]])
derivative_y = sym.diff(y,x)
print("derivative_y \n",derivative_y)



#Find the derivative of the matrix 𝑀=(𝑥2+2𝑥𝑠𝑖𝑛(𝑥)𝑐𝑜𝑠(𝑥)𝑥3+2𝑥) w.r.t. a scalar 𝑥.
y = sym.Matrix([[ x * x + 2 * x, sym.cos(x)],[sym.sin(x), x * x * x + 2 * x ]])
derivative_y = sym.diff(y,x)
print("derivative_y \n",derivative_y)

#To test for functional dependence (both linear and non linear equations) between different equations we use Jacobian determinant shown by |j|. The Jacobian matrix is the first order derivatives of a vector valued function. Vector valued functions are defined as 𝑓:ℝ𝑛→ℝ𝑚.
# Given 𝑥∈ℝ𝑛 and 𝑓𝑗:ℝ𝑛→ℝ we have
# 𝑓(𝑥)=⎡⎣⎢⎢⎢⎢𝑓1(𝑥)𝑓2(𝑥)⋮𝑓𝑚(𝑥)⎤⎦⎥⎥⎥⎥

# We could then define the jaccobian as
# 𝐽(𝑥)=⎡⎣⎢⎢⎢⎢⎢⎢∂𝑓1∂𝑥1∂𝑓2∂𝑥1⋮∂𝑓𝑚∂𝑥1∂𝑓1∂𝑥2∂𝑓2∂𝑥2⋮∂𝑓𝑚∂𝑥2……⋱…∂𝑓1∂𝑥𝑛∂𝑓2∂𝑥𝑛⋮∂𝑓𝑚∂𝑥𝑛⎤⎦⎥⎥⎥⎥⎥⎥

#Hence each row represents the derivative of a real valued function with input vectors. Note that using shape convention we must reshape that to have the same output as the vector input. 

#Let's find the derviate of function 𝑓(𝑥,𝑦)=𝑥+𝑠𝑖𝑛(𝑦) with respect to x and y
# Defining x and y as symbols
x, y = sym.symbols("x, y") 

# Defining the function f(x, y)
f = x + sym.sin(y) 
print(f)

# Derivate w.r.t to 'x'
derivative_fx = f.diff(x)

# Derivate w.r.t to 'y'
derivative_fy = f.diff(y) 

# If we organize these partials into a horizontal vector, we get the gradient of 𝑓(𝑥,𝑦), or Δ𝑓(𝑥,𝑦) :
# Δ𝑓(𝑥,𝑦)=[∂𝑓(𝑥,𝑦)∂𝑥,∂𝑓(𝑥,𝑦)∂𝑦]=[1,𝑐𝑜𝑠(𝑦)]
gradient_fxy = [derivative_fx, derivative_fy]
print("The gradient of the given function is:", gradient_fxy)


#Now let's find the derivative of function 𝑔(𝑥,𝑦)=𝑠𝑖𝑛(𝑥)+𝑦 with respect to x and y
# Defining the function g(x, y)
g = sym.sin(x) + y 
print("The given function is:", g)
# Derivate w.r.t to 'x'
derivative_gx = g.diff(x) 

# Derivate w.r.t to 'y'
derivative_gy = g.diff(y) 

print(derivative_gx)
print(derivative_gy)

# If we organize these partials into a horizontal vector, we get the gradient of 𝑔(𝑥,𝑦), or Δ𝑔(𝑥,𝑦):
#Δ𝑔(𝑥,𝑦)=[∂𝑔(𝑥,𝑦)∂𝑥,∂𝑔(𝑥,𝑦)∂𝑦]=[𝑐𝑜𝑠(𝑥),1]

gradient_gxy = [derivative_gx, derivative_gy]
print("The gradient of the given function is:", gradient_gxy)

# If we organize the gradients of 𝑓(𝑥,𝑦) and 𝑔(𝑥,𝑦) into a single matrix, we move from vector calculus into matrix calculus. This matrix, and organization of the gradients of multiple functions with multiple variables, is known as the Jacobian matrix.
# 𝐽=[Δ𝑓(𝑥,𝑦)Δ𝑔(𝑥,𝑦)]=⎡⎣⎢⎢∂𝑓(𝑥,𝑦)∂𝑥∂𝑔(𝑥,𝑦)∂𝑥∂𝑓(𝑥,𝑦)∂𝑦∂𝑔(𝑥,𝑦)∂𝑦⎤⎦⎥⎥=[1𝑐𝑜𝑠(𝑥)𝑐𝑜𝑠(𝑦)1]

jacobian_matrix = sym.Matrix([gradient_fxy, gradient_gxy])
print("The Jacobian Matrix is:", jacobian_matrix) 

# Numerator Layout Notation:
#This approach is used when we have to find the derivative w.r.t. a vector. Using Numerator Layout Notation, we find the partial derivative of a scalar, matrix, or a vector w.r.t. each element of the vector.
#Suppose, there is a vector 𝑥⃗  = (𝑥1, 𝑥2, 𝑥3,.... 𝑥𝑛) w.r.t. which we have to find the derivative of a scalar function or a matrix.
#For this, we will take the partial derivative of the scalar function or a matrix w.r.t. each element of the vector as shown below:
#∂∂𝑥 = (∂∂(𝑥1),∂∂(𝑥2),∂∂(𝑥3))


# Dervatives w.r.t. Vector
#Scalar by Vector

#
#    Given a scalar function 𝑦=tan(𝑥2+𝑦𝑧). Find the derivative of 𝑦 w.r.t. the below vector:
#
#    𝑣⃗ (𝑥,𝑦,𝑧) = 𝑥𝑦𝑧
#
#    Hint: Apply numerator layout notation where we take derivative of the function w.r.t. every element of the vector.

# Defining symbols that will be used as variables in the function
z = sym.Symbol('z')

# Defining a scalar function using the symbols x, y and z
scalar_f = sym.tan(x * x + y * z)

# Differentiating the scalar function w.r.t. each element of the vector i.e. x, y and z
dx, dy, dz = scalar_f.diff(x), scalar_f.diff(y), scalar_f.diff(z) 
dx, dy, dz


## Find the derivative of the below matrix 𝑀 w.r.t. vector 𝑣⃗  = 𝑥𝑦𝑧:
#(𝑥+𝑥2𝑧𝑥𝑦𝑧𝑐𝑜𝑠𝑥3𝑦+𝑥2)

# Defining the matrix M
M =  sym.Matrix([[x + x * x * z, sym.cos(x)], [x * y * z, 3 * y + x * x]])

# Differentiating matrix 'M' w.r.t. each element of the vector v that are x, y, z
derivative_mx = M.diff(x)
derivative_my = M.diff(y)
derivative_mz = M.diff(z)

# Result -> Organizing the result of partial derivatives into a single matrix
diff_result = sym.Matrix([derivative_mx, derivative_my, derivative_mz])
diff_result

# Find the derivative of vector 𝑣⃗  = ⎛⎝⎜⎜⎜𝑐𝑜𝑠𝑥+𝑥𝑥2+𝑦23𝑥3−𝑧2𝑦⎞⎠⎟⎟⎟ w.r.t. a vector 𝑘⃗  = 𝑥𝑦𝑧

# Defining the vector v 
v =  sym.Matrix([[sym.cos(x) + x], [x * x + y * y], [3 * x * x * x - z * z * y]])

# Vector k = xyz indicates that we have to differentiate vector 'v' w.r.t. x, y, and z
derivative_vx = v.diff(x)
derivative_vy = v.diff(y)
derivative_vz = v.diff(z) 

# Result -> Organizing the result of partial derivatives into a single matrix
diff_v_result = sym.Matrix([derivative_vx, derivative_vy, derivative_vz])
diff_v_result

# To find the derivative of a scalar w.r.t. a matrix, we have to find the derivative of the scalar function w.r.t. every element of the matrix to get the final result.
# Further, let us understand with th help of a question.
# Find the derivative of a scalar 𝑦=(𝑠𝑖𝑛𝑥1+2𝑥2+𝑥23+7𝑥4) w.r.t. a matrix 𝑀=(𝑥1𝑥3𝑥2𝑥4).
# Defining the symbols for the matrix M
x1 = sym.Symbol('x1')
x2 = sym.Symbol('x2')
x3 = sym.Symbol('x3')
x4 = sym.Symbol('x4')

# Defining a scalar function using the symbols declared above which are used as variables
y = sym.sin(x1) + 2 * x2 + x3 * x3 + 7 * x4

# Defining a matrix
M =  sym.Matrix([[x1,x2],[x3,x4]])

# Differentiation of scalar function w.r.t. every element of the matrix 
dx1 = y.diff(x1)
dx2 = y.diff(x2)
dx3 = y.diff(x3)
dx4 = y.diff(x4)

# Organizing the result of partial derivatives into a single matrix
d_result = sym.Matrix([[dx1,dx2],[dx3,dx4]])
d_result

#
# Here, we have to find the derivative of the given matrix w.r.t. each element of the matrix by which we have to differentiate.
# Let us solve a question to understand derivative of a matrix wr.r.t. a matrix.
# Find the derivative of a matrix 𝐴 = (𝑥21+𝑥22+𝑥2𝑠𝑖𝑛𝑥2+𝑥23𝑥22+𝑥24𝑥1+𝑥3+𝑥4) w.r.t. a matrix 𝑀=(𝑥1𝑥3𝑥2𝑥4).

# Defining the matrix A
A =  sym.Matrix([[x1 * x1 + x2 * x2 + x2, x2 * x2 + x4 * x4],[sym.sin(x2) + x3 * x3, x1 + x3 + x4]])

# Defining matrix M
M = sym.Matrix([[x1,x2],[x3,x4]])

# Differentiating matrix A w.r.t. each element of matrix M
dx1 = A.diff(x1)
dx2 = A.diff(x2)
dx3 = A.diff(x3)
dx4 = A.diff(x4)

# Organizing the result of partial derivatives into a single matrix
mm_result = sym.Matrix([[dx1,dx2],[dx3,dx4]])
mm_result

#As vector is a special case of matrix, so, we will consider the vector as a matrix while finding the derivative using Python.
#Find the derivative of vector 𝑚⃗  = ⎛⎝⎜⎜⎜𝑠𝑖𝑛𝑥1+𝑥2𝑥23+𝑥222𝑥2+𝑥4⎞⎠⎟⎟⎟ w.r.t. the matrix 𝑀=(𝑥1𝑥3𝑥2𝑥4).
# Defining vector v_new
v_new =  sym.Matrix([sym.sin(x1)+x2,x3*x3+x2*x2,2*x2+x4])

# Defining matrix M_new
M_new = sym.Matrix([[x1,x2],[x3,x4]])

# Differentiating vector v_new w.r.t. each element of matrix M_new
dx1 = v_new.diff(x1)
dx2 = v_new.diff(x2)
dx3 = v_new.diff(x3)
dx4 = v_new.diff(x4)

# Organizing the result of partial derivatives into a single matrix
vm_result = sym.Matrix([[dx1,dx2],[dx3,dx4]])
vm_result




