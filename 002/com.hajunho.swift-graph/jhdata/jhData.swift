//
//  jhData.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 19..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

struct jhData {
    
    internal var mValuesOfDatas : Array<CGFloat> = Array() {
        didSet {
            if GS.shared.logLevel.contains(.graph) {
                print("mValuesOfDatas.count has been changed to \(mValuesOfDatas.count) in jhData")
            }
        }
    }
    
    //calculated property related with DATAs' View
    private var mAllofCountOfDatas : Int {
        get {
            return self.mValuesOfDatas.count
        }
    }
    
    var mCountOfDatas : Int
    var mMaxValueOfDatas : CGFloat
    var mMinvalueOfDatas : CGFloat
    
    internal var mVerticalRatioToDraw_view : CGFloat = 1.0
    
    /// Axes
    var mCountOfaxes_view : Int
    var mUnitOfHorizontalAxes : CGFloat
    var mcountOfHorizontalAxes : Int
    
    var axisDistance : CGFloat {
        get {
            return (jhDraw.maxR  - mMargin * 2) / CGFloat(mCountOfaxes_view+1)
        }
//        set(distance) {
//            mCountOfaxes_view = Int(jhDraw.maxR  / CGFloat(distance))
//        }
    }
    
    internal var mMargin : CGFloat = 300 //1000.0
    //1000.0 is 13.3..%, margin between panel & graph area 0<=martgin<10000.0
    
    init() {
        mCountOfDatas = 0
        mMaxValueOfDatas = 0
        mMinvalueOfDatas = 0
        
        mCountOfaxes_view = 1
        mUnitOfHorizontalAxes = 100
        mcountOfHorizontalAxes = 3
        //This will be moved to jhScene
        
        let dataSource = jhFile.legacyConverterToArray("testdata", "plist")!
        
        var maxValue : CGFloat = 0.0
        var minValue : CGFloat = jhDraw.maxR
        
        for element in dataSource {
            let _element = element as! NSArray
            let vDate = _element[0] as! CFDate
            let vNumber = _element[1] as! CGFloat
            
            if GS.shared.logLevel.contains(.graph) {
                print("datasrc2 \(vDate) \(vNumber)")
            }
            
            if vNumber > maxValue { maxValue = vNumber }
            if vNumber < minValue { minValue = maxValue }
            mValuesOfDatas.append(vNumber) //TODO:
            jhClientServer.mValuesOfDatas.append(vNumber)
            mCountOfDatas = mValuesOfDatas.count
        }
        
        mMaxValueOfDatas = maxValue
        mMinvalueOfDatas = minValue
        
        mVerticalRatioToDraw_view = (jhDraw.maxR - (2*mMargin)) / mMaxValueOfDatas
        if GS.shared.logLevel.contains(.graph) {
            print("mVerticalRatioToDraw_view =", mVerticalRatioToDraw_view)
            
        }
        mCountOfaxes_view = mAllofCountOfDatas
    }
    
//    func getArrayOfData() -> NSArray {
//        return jhFile.legacyConverterToArray("testdata", "plist")!
//    }
    
}
