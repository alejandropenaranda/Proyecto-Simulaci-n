import numpy as np

board = [[1,0,0,1],
         [1,0,0,1],
         [1,1,1,1],
         [1,0,0,1],
         [1,0,0,1]]

A = [[-8,0,1,0,0,0,0,0,0,0,0,0],
     [0,-8,0,1,0,0,0,0,0,0,0,0],
     [3,0,-8,0,1,0,0,0,0,0,0,0],
     [0,3,0,-8,0,0,0,1,0,0,0,0],
     [0,0,3,0,-8,1,0,0,1,0,0,0],
     [0,0,0,0,3,-8,1,0,0,0,0,0],
     [0,0,0,0,0,3,-8,1,0,0,0,0],
     [0,0,0,3,0,0,3,-8,0,1,0,0],
     [0,0,0,0,3,0,0,0,-8,0,1,0],
     [0,0,0,0,0,0,0,3,0,-8,0,1],
     [0,0,0,0,0,0,0,0,3,0,-8,0],
     [0,0,0,0,0,0,0,0,0,3,0,-8]]

bx = [[-3],
      [0],
      [-3],
      [0],
      [-3],
      [0],
      [0],
      [0],
      [-3],
      [0],
      [-3],
      [0]]

by = [[0],
      [0],
      [0],
      [0],
      [0],
      [0],
      [0],
      [0],
      [0],
      [0],
      [0],
      [0]]

x = np.zeros ((12,1)) # solucion inicial

def Jacobi(A,b,x,n,tol): # n es numero de iteraciones
    D = np.diag(A).reshape(x.shape)
    R = A - np.diagflat(D)
    for i in range(n):
        xi = x
        x = (b - np.dot(R,x))/D
        tolx = np.linalg.norm(x-xi)/np.linalg.norm(x)
        if tolx<tol:
            break
    print('numero de iteraciones',i)
    print('tol J',tolx)    
    return x

print( 'resultado Jaco',Jacobi(A,bx,x,50,0.00000000001))

def Jacobi_Sobre_Relajacion(A,b,x,n,tol,omega): # n es numero de iteraciones, omega debe ser mayor que 1 para sobreRelajación
    D = np.diag(A).reshape(x.shape)
    R = A - np.diagflat(D)
    for i in range(n):
        xi = x
        x = (1-omega)*xi + omega*(b - np.dot(R,x))/D
        tolx = np.linalg.norm(x-xi)/np.linalg.norm(x)
        if tolx<tol:
            break
    print('numero de iteraciones',i)
    print('tol JS',tolx)
    return x

print('resultado Jacos',Jacobi_Sobre_Relajacion(A,bx,x,50,0.00000000001,1.013))

