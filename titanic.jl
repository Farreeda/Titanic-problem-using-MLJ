using DataFrames
using DataFramesMeta
using CSV, Plots
using MLJ
using MLJBase
#, MLDataUtils
using EvoTrees

train_df = CSV.read("data/train.csv", DataFrame)
test_df = CSV.read("data/test.csv",DataFrame)
describe(train_df)
train_df = dropmissing(train_df,"Embarked")
test_df = dropmissing(test_df,"Embarked")
train_df.Age = replace(train_df.Age,missing=>28)
test_df.Age = replace(test_df.Age,missing=>28)
train_df = select!(train_df, Not("Cabin"))
test_df = select!(test_df,Not(["Cabin","PassengerId","Name"]))
train_df = select!(train_df,Not(["PassengerId","Name"]));
combine(groupby(train_df,"Embarked"),nrow=>"count")
train_df.Embarked = Int64.(
    replace(train_df.Embarked, 
        "S" => 1, "C" => 2, "Q" => 3
    )
)
combine(groupby(test_df,"Embarked"),nrow=>"count")
test_df.Embarked = Int64.(
    replace(test_df.Embarked, 
        "S" => 1, "C" => 2, "Q" => 3
    )
)
train_df.Sex = Int32.(replace(train_df.Sex, "female" => 0, "male" => 1))
test_df.Sex = Int32.(replace(test_df.Sex, "female" => 0, "male" => 1))
combine(groupby(train_df,"Ticket"),nrow=>"count")
train_df = select!(train_df, Not("Ticket"))
combine(groupby(test_df,"Ticket"),nrow=>"count")
test_df = select!(test_df, Not("Ticket"))

y,X= unpack(train_df, ==(:Survived),colname->true)
train, test = partition(eachindex(y), 0.7, stratify=y)
describe(train_df)
#--------------------------------------------
#first(Xtrain,4)|>pretty
#models(matching(Xtrain,ytrain))
#doc("EvoTreeCount")
#-----------------------------------------------------------------------
#EvoTreeCount = @load EvoTreeCount pkg=EvoTrees
#Evomodel = EvoTreeCount(max_depth=5, nbins=32, nrounds=100)
#mach = machine(model, Xtrain, ytrain) |> fit!
#preds_test = predict(mach, test_df)

#EvoTrees.fit_evotree(Evomodel, x_train=Matrix{Float32}(X[train,:]), y_train=y[train])
#Evoŷ=EvoTrees.predict!(Evomodel,Matrix{Float32}(X[test,:]))
#---------------------------------------------------------

@load DecisionTreeClassifier pkg="DecisionTree"
model = DecisionTreeClassifier(max_depth=3)
DecisionTree.fit!(model, Matrix(X[train,:]), y[train])
ŷ = DecisionTree.predict(model, Matrix(X[test, :]))
accuracy_score = MLJBase.accuracy(ŷ, y[test])

#-----------------------------------
NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg="MLJFlux"
model2= NeuralNetworkClassifier()

