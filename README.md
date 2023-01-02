# Training Model Using Playgrounds

```swift
import Cocoa
import CreateML
import Foundation

let filePath = "/Users/yamamotokyou/Desktop/epinions3.csv"
let outputFileName = "/Users/yamamotokyou/Desktop/SentimentAnalysisClassiferModel.mlmodel"

let dataURL = URL(fileURLWithPath: filePath)
let data = try MLDataTable(contentsOf: dataURL)

let (trainingData,testingData) = data.randomSplit(by: 0.8)

let sentimentClassifier = try MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "class")

let metadata = MLModelMetadata(author: "Kyo Yamamoto", shortDescription: "Sentiment Analysis Model", version: "1.0")

try sentimentClassifier.write(to: URL(fileURLWithPath: outputFileName), metadata: metadata)


```
