//
//  jhType1graph.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 25..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

class jhType1graphPanel<T> : jhPanel<T> {
    override func drawDatas() {
        print("hjh", xDistance)
        //        jhDataCenter.mCountOfdatas_view = mAllofCountOfDatas
        
        dataLayer = jhType1graphLayer<T>(self, 0)
        
        dataLayer.frame = CGRect(x: 0, y: 0, width: self.mPanelWidth!, height: self.mPanelHeight!) //TODO: will be changed.
        dataLayer.zPosition=1
        //        guideLine.isGeometryFlipped = true
        dataLayer.backgroundColor = UIColor(white: 1, alpha:0.5).cgColor
        self.layer.addSublayer(dataLayer)
        dataLayer.setNeedsDisplay()
        jhDataCenter.attachObserver(observer: self)
    }
    
    override func jhRedraw() {
        print("hjh", xDistance)
        drawAxes()
        
        if isFixedAxesCount {
            jhDataCenter.mCountOfaxes_view = fixedAxesCount
        } else {
            jhDataCenter.mCountOfaxes_view = mAllofCountOfDatas
        }
        
        jhDataCenter.mCountOfdatas_view = mAllofCountOfDatas
        
        dataLayer = jhType1graphLayer(self, 0)
        
        dataLayer.frame = CGRect(x: 0, y: 0, width: self.mPanelWidth!, height: self.mPanelHeight!) //TODO: will be changed.
        dataLayer.zPosition=1
        //        guideLine.isGeometryFlipped = true
        dataLayer.backgroundColor = UIColor(white: 1, alpha:0.5).cgColor
        self.layer.addSublayer(dataLayer)
        dataLayer.setNeedsDisplay()
        jhDataCenter.attachObserver(observer: self)
        
        drawAxes()
    }
}
