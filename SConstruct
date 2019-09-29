
import os

debug     = False  # activates debugging options
warn      = False  # activates compiler warning
fullmsg   = False  # activates full compilation output

CXX       = 'g++'
CXXFLAGS  = '-std=c++17'

DBGFLAGS  = '-g'
OPTFLAGS  = '-O3 -march=native'
WARNFLAGS = '-Wall'
LINKFLAGS = ''

ARPACK_LFLAGS  = '-larpack'

HLIBPRO_PATH   = '/home/user/hlibpro/main'
INSTALL_PREFIX = '.'  # change to install into other directory

######################################################################
#
# set up compilation environment
#
######################################################################

if debug :
    CXXFLAGS  = CXXFLAGS  + ' ' + DBGFLAGS
    LINKFLAGS = LINKFLAGS + ' ' + DBGFLAGS
else :
    CXXFLAGS  = CXXFLAGS  + ' ' + OPTFLAGS

if warn :
    CXXFLAGS  = CXXFLAGS  + ' ' + WARNFLAGS

env = Environment( ENV        = os.environ,
                   CXX        = CXX,
                   CXXFLAGS   = Split( CXXFLAGS  ),
                   LINKFLAGS  = Split( LINKFLAGS ),
                   )

if not fullmsg :
    env.Replace( CCCOMSTR   = " CC     $SOURCES" )
    env.Replace( CXXCOMSTR  = " C++    $SOURCES" )
    env.Replace( LINKCOMSTR = " Link   $TARGET"  )

env.Append(  CPPPATH = [ 'include' ] )
env.Prepend( LIBPATH = [ "lib" ] )

# add HLIBpro and ARPACK flags
env.ParseConfig( os.path.join( HLIBPRO_PATH, 'bin/hlib-config' ) + ' --cflags --lflags' )
env.MergeFlags( ARPACK_LFLAGS )

hamls = env.StaticLibrary( 'hamls', [ 'src/eigen_misc.cc',
                                      'src/TEigenAnalysis.cc',
                                      'src/TEigenArnoldi.cc',
                                      'src/TEigenArpack.cc',
                                      'src/TEigenLapack.cc',
                                      'src/THAMLS.cc',
                                      'src/TSeparatorTree.cc' ] )

env.Prepend( LIBS    = [ "hamls" ] )
env.Prepend( LIBPATH = [ "." ] )
example = env.Program( 'example/eigensolver.cc' )


#
# installation
#

HEADERS = [ 'include/hamls/arpack.hh',
            'include/hamls/eigen_misc.hh',
            'include/hamls/TEigenAnalysis.hh',
            'include/hamls/TEigenArnoldi.hh',
            'include/hamls/TEigenArpack.hh',
            'include/hamls/TEigenLapack.hh',
            'include/hamls/THAMLS.hh',
            'include/hamls/TSeparatorTree.hh' ]

if INSTALL_PREFIX != '.' :
    for header in HEADERS :
        env.Install( os.path.join( INSTALL_PREFIX, 'include', 'hamls' ), header )

env.Install( os.path.join( INSTALL_PREFIX, 'lib' ), hamls )
env.Install( os.path.join( INSTALL_PREFIX, 'bin' ), example )

env.Alias( 'install', INSTALL_PREFIX )
