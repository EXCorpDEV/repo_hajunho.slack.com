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
                print("mValuesOfDatas.count has been changed to \(mValuesOfDatas.count) in jhPanel")
            }
        }
    }
    
    /// Axes
    var mCountOfaxes_view : Int = 1
    var mUnitOfHorizontalAxes : CGFloat = 100
    var mcountOfHorizontalAxes : Int = 3
    
    internal var mVerticalRatioToDraw_view : CGFloat = 1.0
    
    var axisDistance : CGFloat {
        get {
            return (jhDraw.maxR  - mMargin * 2) / CGFloat(mCountOfaxes_view+1)
        }
        set(distance) {
            mCountOfaxes_view = Int(jhDraw.maxR  / CGFloat(distance))
        }
    }

    var mCountOfDatas : Int
    var mMaxValueOfDatas : CGFloat
    var mMinvalueOfDatas : CGFloat
    
    internal var mMargin : CGFloat = 300 //1000.0
    //1000.0 is 13.3..%, margin between panel & graph area 0<=martgin<10000.0
    
    
    init() {
        mCountOfDatas = 0
        mMaxValueOfDatas = 0
        mMinvalueOfDatas = 0
    }

}
