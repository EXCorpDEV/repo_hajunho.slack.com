//
//  jhBarGraph.swift
//  bridge8
//
//  Created by Junho HA on 2022. 2. 22
//  Copyright © 2022년 eoflow. All rights reserved.
//

import UIKit

class jhBarGraph<T> : jhPanel<T> {
    override func drawDatas() {
        print("hjh", xDistance)
//        jhDataCenter.mCountOfdatas_view = mAllofCountOfDatas
        
        dataLayer = jhBarGraphLayer<T>(self, 0, 400)
        
        dataLayer.frame = CGRect(x: 0, y: 0, width: self.bounds.width, height: self.bounds.height) //TODO: will be changed.
        dataLayer.zPosition=1
        //        guideLine.isGeometryFlipped = true
        dataLayer.backgroundColor = UIColor(white: 1, alpha:0.5).cgColor
        self.layer.addSublayer(dataLayer)
        dataLayer.setNeedsDisplay()
        jhDataCenter2.attachObserver(observer: self)
    }
    
    override func jhRedraw() {
        print("hjh", xDistance)
           drawAxes()

        
        if isFixedAxesCount {
            jhDataCenter2.mCountOfaxes_view = fixedAxesCount
        } else {
            jhDataCenter2.mCountOfaxes_view = mAllofCountOfDatas
        }
        
        jhDataCenter2.mCountOfdatas_view = mAllofCountOfDatas
        
        dataLayer = jhBarGraphLayer(self, 0, 400)
        
        dataLayer.frame = CGRect(x: 0, y: 0, width: self.bounds.width, height: self.bounds.height) //TODO: will be changed.
        dataLayer.zPosition=1
        //        guideLine.isGeometryFlipped = true
        dataLayer.backgroundColor = UIColor(white: 1, alpha:0.5).cgColor
        self.layer.addSublayer(dataLayer)
        dataLayer.setNeedsDisplay()
        jhDataCenter2.attachObserver(observer: self)
        
        drawAxes()
    }
}
